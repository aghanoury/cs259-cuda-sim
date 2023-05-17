import os
import sys
import numpy as np
import argparse

# please feel free to change these parameters as you see fit
# naming convention, style, etc... it's all up to you

# random arguments courtesy of copilot
parser = argparse.ArgumentParser(description="Model parameters")
parser.add_argument("--model", type=str, default="alexnet")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--keep_prob", type=float, default=0.8)
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--n_classes", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--dimx", type=int, default=256)
parser.add_argument("--dimy", type=int, default=256)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--padding", type=int, default=3)

args = parser.parse_args()


# some random parameters that we may want to do something with
kernel_parameters = {
    "x": 5,
    "y": 5,
    "depth": 32,
    "padding": 3,
    "stridex": 1,
    "stridey": 1,
}

layer_parameters = {
    1: {
        "type": "conv",
        "dimx": 256,
        "dimy": 256,
        "depth": 3,
    },
    2: {
        "type": "conv",
        "dimx": 128,
        "dimy": 128,
        "depth": 32,
    },
    3: {
        "type": "pool",
        "dimx": 64,
        "dimy": 64,
        "depth": 32,
    },
    # you get the idea, etc...
}

time_units = 1e-9  # 1 ns
system_latency_parameters = {
    "multiply": 1,
    "add": 1,
    "scratchpad_mem_access": 1,
    "local_mem_access": 1,
}

# literally some random garbage courtesy of copilot - this is not a real model and we probably don't need it
other_parameters = {
    "learning_rate": 0.01,
    "batch_size": 128,
    "n_epochs": 30,
    "n_train": 60000,
    "n_test": 10000,
    "keep_prob": 0.8,
    "n_hidden": 128,
    "n_classes": 10,
}

layer1_ops = (
    layer_parameters[1]["dimx"]
    / kernel_parameters["stridex"]
    * layer_parameters[1]["dimy"]
    / kernel_parameters["stridey"]
    * layer_parameters[1]["depth"]
    * kernel_parameters["x"]
    * kernel_parameters["y"]
    * kernel_parameters["depth"]
)
ideal_execution_time = layer1_ops * \
    system_latency_parameters["multiply"] * time_units
print("Calculations required for layer 1: ", layer1_ops)
print("total execution time: {} ns".format(round(ideal_execution_time, 4)))

# ---------- Mechanistic Model ----------- #

# pseudo performance calc  based on a mechanistic model
# the calculate_performance function takes the kernel parameters,
# layer parameters, and system latency parameters as inputs.
# It iterates over each layer, performs the necessary computations based on the layer type, and accumulates the total number of operations (total_ops).
# The ideal execution time is calculated by multiplying the total number of operations with the system latency parameter.


def calculate_performance(kernel_parameters, layer_parameters, system_latency_parameters):
    # Perform computations for each layer
    total_ops = 0
    total_execution_time = 0

    for layer in layer_parameters:
        layer_params = layer_parameters[layer]
        if layer_params["type"] == "conv":
            ops = (
                layer_params["dimx"]
                / kernel_parameters["stridex"]
                * layer_params["dimy"]
                / kernel_parameters["stridey"]
                * layer_params["depth"]
                * kernel_parameters["x"]
                * kernel_parameters["y"]
                * kernel_parameters["depth"]
            )
        elif layer_params["type"] == "pool":
            ops = 0  # Perform calculations specific to pooling layers
            # Add the necessary equations for pooling layer computations

        total_ops += ops

    # Calculate the ideal execution time
    ideal_execution_time = total_ops * system_latency_parameters["multiply"]

    return total_ops, ideal_execution_time


# Random parameters
kernel_parameters = {
    "x": 5,
    "y": 5,
    "depth": 32,
    "padding": 3,
    "stridex": 1,
    "stridey": 1,
}

layer_parameters = {
    1: {
        "type": "conv",
        "dimx": 256,
        "dimy": 256,
        "depth": 3,
    },
    2: {
        "type": "conv",
        "dimx": 128,
        "dimy": 128,
        "depth": 32,
    },
    3: {
        "type": "pool",
        "dimx": 64,
        "dimy": 64,
        "depth": 32,
    },
    # Add more layer parameters as needed
}

system_latency_parameters = {
    "multiply": 1,
    "add": 1,
    "scratchpad_mem_access": 1,
    "local_mem_access": 1,
}

# Calculate performance metrics
total_ops, ideal_execution_time = calculate_performance(
    kernel_parameters, layer_parameters, system_latency_parameters
)

# Print the results
print("Total operations: ", total_ops)
print("Ideal execution time: {} ns".format(round(ideal_execution_time, 4)))


# --------- ROOFLINE MODEL ------------ #
# pseudo performance calc  based on a roofline model
# function calculates the total number of operations (total_ops) as before.
# However, instead of using the system_latency_parameters directly for execution time,
# it divides the total number of operations by the roofline performance parameter (multiply) to obtain the compute-bound execution time (compute_bound_time).
# Additionally, it calculates the memory-bound execution time (memory_bound_time) by dividing the total number of operations by the scratchpad memory access latency (scratchpad_mem_access).
# The model prediction, ideal_execution_time, is determined by taking the minimum value between compute_bound_time and memory_bound_time.
def calculate_performance(kernel_parameters, layer_parameters, system_latency_parameters):
    # Perform computations for each layer
    total_ops = 0
    total_execution_time = 0

    for layer in layer_parameters:
        layer_params = layer_parameters[layer]
        if layer_params["type"] == "conv":
            ops = (
                layer_params["dimx"]
                / kernel_parameters["stridex"]
                * layer_params["dimy"]
                / kernel_parameters["stridey"]
                * layer_params["depth"]
                * kernel_parameters["x"]
                * kernel_parameters["y"]
                * kernel_parameters["depth"]
            )
        elif layer_params["type"] == "pool":
            ops = 0  # Perform calculations specific to pooling layers
            # Add the necessary equations for pooling layer computations

        total_ops += ops

    # Calculate the roofline performance bounds
    roofline_ops = system_latency_parameters["multiply"]
    compute_bound_time = total_ops / roofline_ops
    memory_bound_time = total_ops / \
        system_latency_parameters["scratchpad_mem_access"]

    # Choose the minimum execution time as the model prediction
    ideal_execution_time = min(compute_bound_time, memory_bound_time)

    return total_ops, ideal_execution_time


# Random parameters
kernel_parameters = {
    "x": 5,
    "y": 5,
    "depth": 32,
    "padding": 3,
    "stridex": 1,
    "stridey": 1,
}

layer_parameters = {
    1: {
        "type": "conv",
        "dimx": 256,
        "dimy": 256,
        "depth": 3,
    },
    2: {
        "type": "conv",
        "dimx": 128,
        "dimy": 128,
        "depth": 32,
    },
    3: {
        "type": "pool",
        "dimx": 64,
        "dimy": 64,
        "depth": 32,
    },
    # Add more layer parameters as needed
}

system_latency_parameters = {
    "multiply": 1,  # Ops per cycle
    "scratchpad_mem_access": 1,  # Ops per cycle
}

# Calculate performance metrics
total_ops, ideal_execution_time = calculate_performance(
    kernel_parameters, layer_parameters, system_latency_parameters
)

# Print the results
print("Total operations: ", total_ops)
print("Roofline execution time: {} cycles".format(
    round(ideal_execution_time, 4)))
