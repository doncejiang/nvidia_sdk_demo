#!/usr/bin/env python3
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
"""Prepare a LeNet ONNX graph for the safe plugin sample in TensorRT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import onnx
from onnx import optimizer
import onnx_graphsurgeon as gs
import numpy as np


def pool_to_2x_pool(pool_node, graph):
    """Replace a MaxPool node with a two MaxPool nodes that can be used for safe engine creation.

    Args:
        pool_node (onnx_graphsurgeon.ir.node.Node): The original MaxPool node.
        graph (onnx_graphsurgeon.ir.graph.Graph): The ONNX-GS graph object that shall be
            modified in-place.
    """
    auto_padding = pool_node.attrs.get('auto_pad', None)
    if auto_padding == 'NOTSET':
        del pool_node.attrs['auto_pad']
    in_tensor = pool_node.inputs[0]
    out_tensor = pool_node.outputs[0]

    pool_attrs = dict(kernel_shape=(2, 2), pads=(0, 0, 0, 0), strides=(2, 2))
    pool_node.attrs.update(pool_attrs)
    aux_shape = in_tensor.shape[:2] + [in_tensor.shape[2] // 2, in_tensor.shape[3] // 2]
    aux_tensor = gs.Variable(name='aux_tensor', dtype=np.float32, shape=aux_shape)
    pool_node.outputs = [aux_tensor]
    aux_attrs = dict(kernel_shape=(2, 2), pads=(0, 0, 1, 1), strides=(2, 2))
    aux_pool = gs.Node(name='aux_pool',
                       op='MaxPool',
                       attrs=aux_attrs,
                       inputs=[aux_tensor],
                       outputs=[out_tensor])
    graph.nodes.append(aux_pool)


def pool_to_plugin(pool_node):
    """Replace a MaxPool node with a MaxPoolPlugin node that can use a custom TensorRT plugin.

    Args:
        pool_node (onnx_graphsurgeon.ir.node.Node): The original MaxPool node that is modified
            to a MaxPoolPlugin node with its required fields (in-place).
    """
    auto_padding = pool_node.attrs.get('auto_pad', None)
    if auto_padding == 'NOTSET':
        del pool_node.attrs['auto_pad']
    pool_node.attrs['plugin_version'] = '1'
    pool_node.attrs['plugin_namespace'] = ''
    pool_node.op = 'MaxPoolPlugin'


def fc_to_conv(nodes, graph, new_output_name='Output'):
    """Replace Reshape, MatMul and Add nodes with a single 4x4 Conv node in a graph (in-place).

    Args:
        nodes (list of onnx_graphsurgeon.ir.node.Node): The Reshape, MatMul and Add nodes that
            shall be replaced.
        graph (onnx_graphsurgeon.ir.graph.Graph): The ONNX-GS graph object that shall be
            modified in-place.
    Optional args:
        new_output_name (str): Output name to be used in the modified graph.
            Defaults to 'Output'.
    """
    assert len(nodes) == 3, 'Expected exactly 3 nodes'
    assert nodes[0].op == 'Reshape', 'Expected the first node to be Reshape, got {}.'.format(nodes[0].op)
    reshape_op = nodes[0]
    reshape_input = reshape_op.inputs[0]
    assert nodes[1].op == 'MatMul', 'Expected the second node to be MatMul, got {}.'.format(nodes[1].op)
    matmul_op = nodes[1]
    matmul_weights = matmul_op.inputs[1]

    # Transpose from (C, K) = (256, 10) to (K, C) = (10, 256) format, required for Conv:
    matmul_weights.values = matmul_weights.values.transpose()

    # Instead of reshaping inputs from (N, C, H, W) = (N, 16, 4, 4) to (N, C) = (N, 16*4*4=256),
    # we directly apply a non-padded 4x4 convolution with the same effect:
    matmul_channels_out, matmul_channels_in = matmul_weights.values.shape
    matmul_weights.values = matmul_weights.values.reshape(matmul_channels_out,
                                                          matmul_channels_in // 16, 4, 4)

    assert nodes[2].op == 'Add', 'Expected the third node to be Add, got {}.'.format(nodes[2].op)
    bias_op = nodes[2]
    bias_weights = bias_op.inputs[1]
    # Flatten the constant Add inputs to (C) = (10):
    bias_weights.values = bias_weights.values.flatten()

    # Instead of the original outputs as (N, C) = (N, 10), the Conv outputs
    # will be (N, C, H, W) = (N, 10, 1, 1):
    old_output = bias_op.outputs[0]
    output_shape = old_output.shape + [1, 1]
    new_output = gs.Variable(name=new_output_name, dtype=np.float32, shape=output_shape)

    # Create a new 4x4 Conv node with our pre-processed bias and weights and add it to the graph:
    conv4x4 = gs.Node(op='Conv',
                      inputs=[reshape_input, matmul_weights, bias_weights],
                      outputs=[new_output])
    conv4x4.attrs['kernel_shape'] = (4, 4)
    conv4x4.attrs['pads'] = (0, 0, 0, 0)
    graph.nodes.append(conv4x4)

    # Override the original graph outputs:
    graph.outputs = [new_output]

    # Now that the previous Reshape, MatMul and Add branch results in a dead end (not producing
    # any graph outputs), we can remove these nodes:
    graph.cleanup()


def process_graph(graph, skip_plugin_replacement=False, debug_replacement=False):
    """Process a LeNet graph for the safe plugin sample in TensorRT (in-place, no returns).

    Args:
        graph (onnx_graphsurgeon.ir.graph.Graph): The ONNX-GS graph object that shall be
            modified in-place.
    Optional args:
        skip_plugin_replacement (bool): For debugging purposes, do not introduce any plugin node
            if True. Defaults to False.
        debug_replacement (bool): For debugging purposes, replace second MaxPool node with
            2 MaxPool nodes with stride/kernel_size 2. Defaults to False.
    """
    fc_ops = graph.nodes[-3:]
    fc_to_conv(fc_ops, graph)

    pool_nodes = [node for node in graph.nodes if node.op == 'MaxPool']
    assert len(pool_nodes) == 2, 'Expected exactly 2 MaxPool nodes, got {}'.format(len(pool_nodes))
    second_pool_node = pool_nodes[1]
    if skip_plugin_replacement:
        if debug_replacement:
            pool_to_2x_pool(second_pool_node, graph)
    else:
        pool_to_plugin(second_pool_node)


def prepare_lenet(onnx_in, onnx_out, skip_plugin_replacement=False, debug_replacement=False):
    """Load a LeNet ONNX model, modify it for safe plugin sample in TensorRT and save it.

    Args:
        onnx_in (str): Path to load the original LeNet input ONNX file from.
        onnx_out (str): Path to save the modified LeNet output ONNX file to.
    Optional args:
        skip_plugin_replacement (bool): For debugging purposes, do not introduce any plugin node
            if True. Defaults to False.
        debug_replacement (bool): For debugging purposes, replace second MaxPool node with
            2 MaxPool nodes with stride/kernel_size 2. Defaults to False.
    """
    model = onnx.load(onnx_in)
    # Fuse the Add nodes into upstream Conv nodes in the ONNX graph (for easier graph handling):
    model = optimizer.optimize(model, ['fuse_add_bias_into_conv'])

    # Import the ONNX graph into ONNX-GraphSurgeon:
    graph = gs.import_onnx(model)

    # Fold constants being inputs to Reshape into the Reshape op itself (results in simpler graph):
    graph.fold_constants()

    # Process the ONNX-GraphSurgeon graph:
    process_graph(graph, skip_plugin_replacement, debug_replacement)

    print('Saving modified ONNX model to {}.'.format(onnx_out))
    onnx.save(gs.export_onnx(graph), onnx_out)


def main():
    """Run the ONNX-GraphSurgeon part of the safe plugin sample for TensorRT."""
    parser = ArgumentParser(description='Modifies a LeNet ONNX graph to accept TensorRT plugins.')
    parser.add_argument("-i",
                        "--input",
                        help='Path to load the original LeNet input ONNX file from.',
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        help='Path to save the modified LeNet output ONNX file to.',
                        required=True)
    parser.add_argument("-s",
                        "--skip-plugin",
                        help='Do not replace the MaxPool node with a MaxPoolPlugin node.',
                        action='store_true')
    parser.add_argument("-d",
                        "--safe-debug",
                        help='If skip_plugin is set, replace the original second MaxPool node with'
                        ' 2 MaxPool nodes that can be used for safe TensorRT engine creation.',
                        action='store_true')
    args = parser.parse_args()
    if args.safe_debug and not args.skip_plugin:
        raise ValueError('Please only set --safe-debug/-d together with --skip-plugin/-s.')
    prepare_lenet(args.input, args.output, args.skip_plugin, args.safe_debug)


if __name__ == '__main__':
    main()
