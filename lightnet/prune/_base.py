#
#   Base pruning method class
#   Copyright EAVISE
#

from abc import ABC, abstractmethod
import logging
import torch
from .dependency import get_dependency_map, NodeType

__all__ = ['Pruner']
log = logging.getLogger(__name__)


def default_get_parameters(model):
    return model.parameters()


class Pruner(ABC):
    """ Abstract pruning class, which should be inheritted by each different method. |br|
    This class takes care of actually performing the soft- or hard-pruning
    and also updates the optimizer after it deleted channels.

    Args:
        model (torch.nn.Module): model to prune
        input_dimensions (tuple): Input dimensions to the network (width, height, channels) where channels is optional and defaults to 3
        optimizer (torch.optim.Optimizer or None, optional): Optimizer that is used when retraining the network (see Note); default **None**
        manner ("soft" or "hard", optional): Whether to perform soft-pruning (replacing channel values with zero) or hard-pruning (deleting channels); Default **"hard"**
        get_parameters (function, optional): function that takes a model and returns the parameters for the optimizer; Default **model.parameters()**

    Note:
        When performing hard-pruning, the number of parameters of the network changes. |br|
        This means that you need to redefine any object that holds a reference to these parameters.

        When (re-)training a network, this means that you need to recreate a new optimizer, or adapt the existing one.
        By passing an ``optimizer`` to this Pruner class,
        this modification is done for you each time you run the pruner.
        You can also modify which parameters are passed to the optimizer, by passing a ``get_parameters`` function.

        The modifications made to the networks whilst hard-pruning also affect weight loading.
        Lightnet modules thus provide the :func:`~lightnet.network.module.Lightnet.load_pruned` function,
        which allows to correctly adapt a network, according to saved pruned weights.

        These modifications are not necessary for soft-pruning,
        as this technique merely changes the values of the weights.
    """
    def __init__(self, model, input_dimensions, optimizer=None, manner="hard", get_parameters=None):
        self.model = model
        self.optimizer = optimizer
        self.manner = manner
        self._soft_pruned = 0
        self._hard_pruned = 0

        if len(input_dimensions) == 2:
            input_dimensions = (1, 3) + input_dimensions[::-1]
        elif len(input_dimensions) == 3:
            input_dimensions = (1,) + input_dimensions[::-1]
        self.dependencies = get_dependency_map(model, input_dimensions)

        if self.optimizer is None:
            log.warn('Pruner did not get an optimizer, make sure to create a new optimizer each time you train after pruning.')

        if get_parameters is not None:
            self.get_parameters = get_parameters
        else:
            self.get_parameters = default_get_parameters

    def __call__(self, percentage):
        """ Perform pruning.

        Args:
            percentage (float): Percentage of the prunable channels to prune (approximative)

        Returns:
            int: Number of pruned channels
        """
        with torch.no_grad():
            # Delete gradients
            for param in self.model.parameters():
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            for buf in self.model.buffers():
                if hasattr(buf, 'grad') and buf.grad is not None:
                    buf.grad.detach_()
                    buf.grad = None

            # Prune
            if self.manner == "soft":
                self._hard_pruned = 0
                prune_count = self._soft_pruned = self.prune(percentage, self.soft_prune)
            else:
                self._soft_pruned = 0
                prune_count = self._hard_pruned = self.prune(percentage, self.hard_prune)
                self._update_optimizer()

            return prune_count

    @abstractmethod
    def prune(self, percentage, prune_manner):
        """ Pruning implementation. |br|
        This method should be implemented by the different methods, and will be called automatically.

        Args:
            percentage (float): Percentage of the prunable channels to prune (approximative)
            prune_manner (function): Function that is used to actually prune (see Note)

        Return:
            int: Number of actually pruned channels

        Note:
            The prune_manner function takes 2 arguments:

            - dependency : item from the dependency-map ``self.dependencies``
            - filter_list : indexes from the channels to prune

            It will then prune the given indexes from the given dependency convolution.
        """
        pass

    @property
    def prunable_channels(self):
        """ Returns the total number of prunable channels left in the network.

        Return:
            int: total number of prunable channels

        Note:
            This property returns the total amount of channels in the prunable convolutions.
            Note that we never prune the last channel of a convolution and thus cannot prune all channels.
        """
        prunable = 0
        for dependency in self.dependencies.values():
            prunable += dependency.module.out_channels

        return prunable

    @property
    def pruned_channels(self):
        """ Returns the number of pruned channels from the last pruning operation. """
        return self._soft_pruned + self._hard_pruned

    @property
    def soft_pruned_channels(self):
        """ Returns the number of soft pruned channels from the last pruning operation. """
        return self._soft_pruned

    @property
    def hard_pruned_channels(self):
        """ Returns the number of hard pruned channels from the last pruning operation. """
        return self._hard_pruned

    def soft_prune(self, conv_node, filter_list, chain=False):
        """ Soft pruning implementation, passed to the prune() function as ``prune_manner`` """
        # Set output tensors to zero; filter_list does not get modified
        if conv_node.module.groups != 1:
            raise NotImplementedError('Grouped Convolution')

        mask = torch.zeros(conv_node.module.weight.shape[0], dtype=bool)
        mask.scatter_(0, torch.tensor(filter_list), True)

        conv_node.module.weight[mask] = 0
        if conv_node.module.bias is not None:
            conv_node.module.bias[mask] = 0

        # Chain modifications
        if chain:
            for c in conv_node.children:
                self._soft_chain(c, filter_list, conv_node)

    def hard_prune(self, conv_node, filter_list, chain=True):
        """ Hard pruning implementation, passed to the prune() function as ``prune_manner`` """
        # Remove filter_list from output tensor; filter_list does not get modified
        if conv_node.module.groups != 1:
            raise NotImplementedError('Grouped Convolution')

        mask = torch.ones(conv_node.module.weight.shape[0], dtype=bool)
        mask.scatter_(0, torch.tensor(filter_list), False)

        conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight[mask])
        if conv_node.module.bias is not None:
            conv_node.module.bias = torch.nn.Parameter(conv_node.module.bias[mask])
        conv_node.module.out_channels -= len(filter_list)

        # Chain modifications
        if chain:
            for c in conv_node.children:
                self._hard_chain(c, filter_list, conv_node)

    def _soft_chain(self, node, filter_list, parent=None):
        if node.type is NodeType.CONV:
            # Set input tensors to zero; filter_list does not get modified
            if node.module.groups != 1:
                # DW-separable
                mask = torch.zeros(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[mask] = 0
            else:
                mask = torch.zeros(node.module.weight.shape[1], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[:, mask] = 0
        elif node.type is NodeType.BATCHNORM:
            # Set tensors to zero; filter_list does not get modified
            if node.module.weight is not None:
                mask = torch.zeros(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[mask] = 0
                if node.module.bias is not None:
                    node.module.bias[mask] = 0

            if node.module.running_mean is not None:
                mask = torch.zeros(node.module.running_mean.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.running_mean[mask] = 0
                if node.module.running_var is not None:
                    node.module.running_var[mask] = 0
        elif node.type is NodeType.CONCAT:
            # Modify filter list according to concat operation
            for p in node.parents:
                if p == parent:
                    break
                else:
                    num = self._get_num_channels(p)
                    filter_list = [f+num for f in filter_list]

        # Recurse
        for c in node.children:
            self._soft_chain(c, filter_list, node)

    def _hard_chain(self, node, filter_list, parent=None):
        if node.type is NodeType.CONV:
            # Remove filter_list from input tensor; filter_list does not get modified
            if node.module.groups != 1:
                # DW-separable
                mask = torch.ones(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[mask])
                node.module.in_channels -= len(filter_list)
                node.module.out_channels -= len(filter_list)
                node.module.groups -= len(filter_list)
            else:
                mask = torch.ones(node.module.weight.shape[1], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[:, mask])
                node.module.in_channels -= len(filter_list)
        elif node.type is NodeType.BATCHNORM:
            # Remove filter_list from tensor; filter_list does not get modified
            if node.module.weight is not None:
                mask = torch.ones(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[mask])
                if node.module.bias is not None:
                    node.module.bias = torch.nn.Parameter(node.module.bias[mask])

            if node.module.running_mean is not None:
                mask = torch.ones(node.module.running_mean.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.running_mean = node.module.running_mean[mask]
                if node.module.running_var is not None:
                    node.module.running_var = node.module.running_var[mask]

            node.module.num_features -= len(filter_list)
        elif node.type is NodeType.CONCAT:
            # Modify filter list according to concat operation
            for p in node.parents:
                if p == parent:
                    break
                else:
                    num = self._get_num_channels(p)
                    filter_list = [f+num for f in filter_list]

        # Recurse
        for c in node.children:
            self._hard_chain(c, filter_list, node)

    def _get_num_channels(self, node):
        if node.type is NodeType.CONV:
            if node.module.groups != 1:
                return node.module.weight.shape[0]
            else:
                return node.module.weight.shape[1]
        if node.type is NodeType.BATCHNORM:
            return node.module.weight.shape[0]
        if node.type is NodeType.CONCAT:
            return sum(self._get_num_channels(p) for p in parents)
        if node.type is NodeType.ELEMW_OP:
            return self._get_num_channels(node.parents[0])

    def _update_optimizer(self):
        if self.optimizer is not None:
            param_groups = list(self.get_parameters(self.model))

            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

            for old, new in zip(self.optimizer.param_groups, param_groups):
                old.update(new)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        get_param = self.get_parameters.__name__ if hasattr(self.get_parameters, '__name__') else self.get_parameters.__class__.__name__
        return f'{self.__class__.__name__} ({self.model.__class__.__name__}, manner={self.manner}, get_parameters={get_param})'
