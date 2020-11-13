#
#   Dependency list related functions
#   Copyright EAVISE
#
import logging
import tempfile
import onnx
import torch
from ._create import *
from ._tree import *
from ._type import *

__all__ = ['get_dependency_map', 'get_onnx_model', 'print_dependency_map']
log = logging.getLogger(__name__)


def get_dependency_map(model, input_dim):
    """ TODO : add proper docstring. """
    # Get ONNX graph
    path = tempfile.TemporaryFile(prefix='lightnet-prune', suffix='.onnx')
    get_onnx_model(model, input_dim, path)
    path.seek(0)
    onnx_model = onnx.load_model(path, load_external_data=False)
    path.close()

    # Create dependency tree
    dependencies = dict()
    for el in onnx_model.graph.node:
        if el.op_type == 'ATen' and el.attribute[0].s == b'_convolution':
            try:
                name = [i for i in el.input if i.endswith('.weight')][0][:-7]
                module = model
                for p in name.split('.'):
                    module = getattr(module, p)
                if module.groups != 1:
                    raise NotImplementedError(f'ATen: Grouped Convolution (cannot prune)')

                dep = create_node(el, onnx_model.graph, model)
                dependencies[dep.name] = dep
            except NotImplementedError as err:
                log.info(f'Cannot prune [{name}], unimplemented dependency [{err}]')
            except StopIteration as err:
                log.info(f'Cannot prune [{name}], generates output [{err}]')

    # Remove ignored and match with modules
    for dep in dependencies.values():
        for _, node in traverse_depth_first(dep):
            if node.type is NodeType.IGNORE:
                for p in node.parents:
                    idx = p.children.index(node)
                    p.children[idx:idx+1] = node.children
                for c in node.children:
                    idx = c.parents.index(node)
                    c.parents[idx:idx+1] = node.parents
                del node
            elif node.name is not None:
                path = node.name.split('.')
                module = model
                try:
                    for p in path:
                        module = getattr(module, p)
                    node.module = module
                except AttributeError:
                    log.debug(f'Could not get PyTorch module for [{node.name}]')
                    continue

    return dependencies


def get_onnx_model(model, input_dim, path):
    """ TODO : add proper docstring. """
    # Get device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        log.warn('Could not determine device from model, using "cpu".')
        device = torch.device('cpu')

    # Create input tensor
    input_tensor = torch.rand(*input_dim).to(device)

    # Create onnx model
    torch.onnx.export(
        model, input_tensor, path,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
    )


def print_dependency_map(dependencies, separator=' '):
    for name, d in dependencies.items():
        print(name)
        for i, n in traverse_depth_first(d, True):
            print(f' {2 * i * separator}{n}')
        print()