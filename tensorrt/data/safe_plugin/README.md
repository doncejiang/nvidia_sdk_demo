# Model

## Purpose

This model was custom designed by Oliver Knieps and Le An at NVIDIA for use
as an example network that meets the safety restrictions and includes a plugin.
The model does digit recognition trained on the MNIST dataset.

## Operation

The plugin is a simple max pooling plugin that reduces 14x14 data to 4x4 using
a 3x3 kernel with stride 3. This operation is supported in standard with normal
pooling, but the stride of 3 means that it is not directly supported in the
current safety subset.

## Generation

The generation of the ONNX file is done by manipulating an existing LeNet
ONNX file. The input file should be a working LeNet network. The script
`safe_plugin_preprocessing.py` will use Graph Surgeon to manipulate nodes
and update the network appropriately.

## Use

Because it requires a plugin it cannot be run using `trtexec`. It should be built
and run using the `sampleSafePlugin` sample.

The plugin is named `MaxPoolPlugin` and is version `1`.

## Files

mnist_safe_plugin.onnx: ONNX model demonstrating plugins in safety

safe_plugin_preprocessing.py: Pre-processing script to convert LeNet into safe network using plugin

