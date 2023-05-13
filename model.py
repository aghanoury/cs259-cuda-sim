import os
import sys
import numpy as np

# please feel free to change these parameters as you see fit
# naming convention, style, etc... it's all up to you

# some random parameters that we may want to do something with
kernel_parameters = {
    "x": 5,
    "y": 5,
    "depth": 32,
    "padding": 3,
}

layer_parameters = {
    1: {
        "type": "conv",
        "dim_x": 256,
        "dim_y": 256,
        "depth": 3,
    },
    2: {
        "type": "conv",
        "dim_x": 128,
        "dim_y": 128,
        "depth": 32,
    },
    3: {
        "type": "pool",
        "dim_x": 64,
        "dim_y": 64,
        "depth": 32,
    },
# you get the idea, etc...
}

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



if __name__ == "__main__":
    print("do stuff...")