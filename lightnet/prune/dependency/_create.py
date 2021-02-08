#
#   Node creation functions
#   Copyright EAVISE
#
import functools
from .._imports import *
from ._tree import *
from ._tree import traverse_parents_breadth_first
from ._type import *


__all__ = ['create_node', 'create_node_inverse']


# Recursive node creation
def create_node(element, graph, model, rootnode=None, parent=None):
    nodetype = get_node_type(element)
    create, out_dep, in_dep = node_func_map[nodetype]
    node = create(element, graph, model, rootnode, parent)
    deps = out_dep(element, graph, rootnode, node)

    # Check if rootnode
    if rootnode is None:
        rootnode = node

    # Loop through children and recurse
    for out in deps:
        if out in [o.name for o in graph.output]:
            raise StopIteration(out)

        for gn in graph.node:
            if out in gn.input:
                create_node(gn, graph, model, rootnode, node)

    return rootnode


def create_node_inverse(element, graph, model, rootnode, child):
    nodetype = get_node_type(element)
    create, out_dep, in_dep = node_func_map[nodetype]
    node = create(element, graph, model, rootnode, None)
    deps = in_dep(element, graph, rootnode, node)

    # StopNode
    if node is None:
        return

    # Check if node already exists
    if child is not None and node in child.parents:
        return [n for n in child.parents if node == n][0]
    if rootnode is not None:
        for _, c in traverse_depth_first(rootnode):
            if node == c:
                c.add_child(child)
                return c

    # Add child to node
    if child is not None:
        node.add_child(child)

    # Loop through parents and recurse
    for inp in deps:
        for gn in graph.node:
            if inp in gn.output:
                create_node_inverse(gn, graph, model, rootnode, node)

    return node


# Create specific nodes
def create_module_weight(nodetype, element, graph, model, rootnode, parent):
    name = [i for i in element.input if i.endswith('.weight')][0][:-7]
    try:
        module = model
        for p in name.split('.'):
            module = getattr(module, p)
    except AttributeError:
        log.debug(f'Could not get PyTorch module for [{name}]')
        module = None

    node = Node(nodetype, name, metadata=(set(element.input), set(element.output)), module=module)
    if parent is not None:
        parent.add_child(node)

    return node


def create_conv(element, graph, model, rootnode, parent):
    node = create_module_weight(NodeType.CONV, element, graph, model, rootnode, parent)

    if node.module.groups == 1:
        # Regular convolution
        return node
    elif node.module.groups == node.module.in_channels == node.module.out_channels:
        # Depthwise separable convolution
        return node
    else:
        # Grouped convolution
        raise NotImplementedError('ATen: Grouped Convolution (Only supporting DW-separable conv as dependency)')


def create_concat(element, graph, model, rootnode, parent):
    node = Node(NodeType.CONCAT, metadata=(set(element.input), set(element.output)))

    # Get concatenation axis (Constant input to concat operation)
    inputs = [i for i in element.input if len(i) > 0]
    for gn in graph.node:
        if gn.output[0] in inputs and gn.op_type == 'Constant':
            value = onnx_numpy_helper.to_array(onnx.helper.get_attribute_value(gn.attribute[0]))
            if value != 1:
                raise NotImplementedError(f'ATen: cat (Not concatenating on channel dimension [1])')

    deps = input_dep_all(element, graph, rootnode, node)
    for inp in deps:
        for gn in graph.node:
            if inp in gn.output:
                create_node_inverse(gn, graph, model, rootnode, node)

    if parent is not None and parent not in node.parents:
        parent.add_child(node)
    return node


def create_elemw_op(element, graph, model, rootnode, parent):
    attr = onnx.helper.printable_attribute(element.attribute[0])
    attr = attr.split('=')[-1].strip()[1:-1]

    if parent is not None:
        # Parent is None in create_node_inverse and in first iteration of create_node
        # elem_op should never be root of tree (is always conv), but this is probably brittle code
        raise NotImplementedError(f'ATen: {attr} (downward path)')

    node = Node(NodeType.ELEMW_OP, attr, metadata=(set(element.input), set(element.output)))

    # Search for a parent node from which we can get size information
    deps = input_dep_all(element, graph, rootnode, node)
    good_dep = None
    for inp in deps:
        for gn in graph.node:
            if inp in gn.output:
                try:
                    dep_node = create_node_inverse(gn, graph, model, None, None)
                except (NotImplementedError, StopIteration) as err:
                    dep_node = None

                if dep_node:
                    for _, n in traverse_parents_breadth_first(dep_node):
                        if n.type in {NodeType.CONV, NodeType.BATCHNORM, NodeType.CONCAT}:
                            good_dep = n
                            break

                    if good_dep is not None:
                        break
        if good_dep is not None:
            break

    # No suitable parent found
    if good_dep is None:
        raise NotImplementedError(f'ATen: {attr} (no readable upward dependency)')

    good_dep.children = list()
    good_dep.parents = list()
    good_dep.add_child(node)
    return node


def create_ignore(element, graph, model, rootnode, parent):
    name = onnx.helper.printable_attribute(element.attribute[0]).split('=')[-1].strip()
    node = Node(NodeType.IGNORE, name, metadata=(set(element.input), set(element.output)))

    if name == '"flip"':
        inputs = [i for i in element.input if len(i) > 0]
        for gn in graph.node:
            if gn.output[0] in inputs and gn.op_type == 'Constant':
                value = onnx_numpy_helper.to_array(onnx.helper.get_attribute_value(gn.attribute[0]))
                if value == 1:
                    raise NotImplementedError(f'ATen: flip (Flipping on channel dimension [1])')

    if parent is not None:
        parent.add_child(node)

    return node


def create_ignore_stop(element, graph, model, rootnode, parent):
    return None


# Return dependencies
def output_dep_all(element, graph, rootnode, node):
    return element.output


def input_dep_all(element, graph, rootnode, node):
    inputs = [i for i in element.input if len(i) > 0]

    # Remove graph inputs
    for i in graph.input:
        if i.name in inputs:
            inputs.remove(i.name)

    # Remove constants
    for gn in graph.node:
        if gn.output[0] in inputs and gn.op_type == 'Constant':
            inputs.remove(gn.output[0])

    return inputs


def dep_none(element, graph, rootnode, node):
    return list()


def output_dep_conv(element, graph, rootnode, node):
    if rootnode is None or node.module.groups != 1:
        return element.output
    else:
        return list()


# Map with 3 funcs (create, output_dep, input_dep) for each NodeType
# WARNING : create/input_dep cannot count on rootnode being correct, because it is not when constructing tree inversely (cat/elemw)
node_func_map = {
    NodeType.CONV: (
        create_conv,
        output_dep_conv,
        dep_none
    ),
    NodeType.BATCHNORM: (
        functools.partial(create_module_weight, NodeType.BATCHNORM),
        output_dep_all,
        dep_none
    ),
    NodeType.CONCAT: (
        create_concat,
        output_dep_all,
        dep_none,        # Gets input deps in create function
    ),
    NodeType.ELEMW_OP: (
        create_elemw_op,
        dep_none,
        dep_none,        # Gets input deps in create function
    ),
    NodeType.IGNORE: (
        create_ignore,
        output_dep_all,
        input_dep_all
    ),
    NodeType.IGNORE_STOP: (
        create_ignore_stop,
        dep_none,
        dep_none
    ),
}
