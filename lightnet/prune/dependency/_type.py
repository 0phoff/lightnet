#
#   Categorization of graph operations into NodeType's
#   Copyright EAVISE
#
import enum
from .._imports import onnx

__all__ = ['NodeType', 'get_node_type', 'add_nodetype_operation']


class NodeType(enum.Enum):
    """ Enum containing all different types of operation in the dependency graph. """
    CONV = enum.auto()          #: Convolution operation
    BATCHNORM = enum.auto()     #: BatchNorm operation
    CONCAT = enum.auto()        #: Concat operation on channel dimension
    ELEMW_OP = enum.auto()      #: Element-wise operation (eg. add)
    IGNORE = enum.auto()        #: Node that does not need to be modified and does not change output dims (eg. relu)
    IGNORE_STOP = enum.auto()   #: Ignore node and stop dependency chain


def get_node_type(element):
    op_type = element.op_type
    if op_type == 'ATen':
        attr = onnx.helper.printable_attribute(element.attribute[0])
        attr = attr.split('=')[-1].strip()[1:-1]

        for key, value in node_type_map_aten.items():
            if attr in value:
                return key

        raise NotImplementedError(f'ATen: {attr}')
    else:
        for key, value in node_type_map_onnx.items():
            if op_type in value:
                return key

        raise NotImplementedError(element.op_type)


def add_nodetype_operation(name, node_type, aten=True):
    """ Add a graph operation to a specific NodeType. |br|
    Adding all possible ONNX/ATen graph operations to our list is not feasible,
    and thus we test our algorithms only with the models in this library.

    If you get the following error, whilst generating the dependency map of your network,
    it means you use an operation which we have not added to our list of known operations:

    .. code:: bash

       Cannot prune [{layer}], unimplemented dependency [{operation}]

    You can then add your operation to a certain :class:`~lightnet.prune.dependency.NodeType` with this function.
    If the operation name starts with **"ATen:"**,
    add the operation without this prefix and set the ``aten`` argument to **True**.
    Otherwise, set ``aten`` to **False**.

    Args:
        name (str): Name of the operation in the graph (dont forget to remove "ATen:" if necessary)
        node_type (NodeType): To which type to add
        aten (bool, optional): Whether to operation is an ATen or ONNX operation in the graph; Default **True**

    Note:
        We would strongly appreciate it if anyone with unimplemented operations, could open an issue on our gitlab.
        That way, we can manually add it to our list and grow it organically based on usage!
    """
    if aten:
        node_type_map_aten[node_type].add(name)
    else:
        node_type_map_onnx[node_type].add(name)


# Default maps
node_type_map_aten = {
    NodeType.CONV: {
        '_convolution',
    },
    NodeType.BATCHNORM: {
        'batch_norm',
    },
    NodeType.CONCAT: {
        'cat',
    },
    NodeType.ELEMW_OP: {
        'add',
    },
    NodeType.IGNORE: {
        'leaky_relu',
        'relu',
        'hardtanh',
        'adaptive_avg_pool2d',
        'max_pool2d',
        'cummax',
        'upsample_nearest2d',
        'replication_pad2d',
        'flip',
    },
    NodeType.IGNORE_STOP: {
        'size',
    },
}


node_type_map_onnx = {
    NodeType.CONV: set(),
    NodeType.BATCHNORM: set(),
    NodeType.CONCAT: set(),
    NodeType.ELEMW_OP: set(),
    NodeType.IGNORE: set(),
    NodeType.IGNORE_STOP: {
        'Concat',
    },
}
