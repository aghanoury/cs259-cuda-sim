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


if __name__ == "__main__":
    # print something

    total_ops = (
        layer_parameters[1]["dimx"]
        / kernel_parameters["stridex"]
        * layer_parameters[1]["dimy"]
        / kernel_parameters["stridey"]
        * layer_parameters[1]["depth"]
        * kernel_parameters["x"]
        * kernel_parameters["y"]
        * kernel_parameters["depth"]
    )
    ideal_execution_time = (
        total_ops * system_latency_parameters["multiply"] * time_units
    )
    print("Calculations required for layer 1: ", total_ops)
    print("total execution time: {} ns".format(round(ideal_execution_time, 4)))


# w	h	c	n (elem / batch)	k	f_w	f_h	pad_w	pad_h	stride_w	stride_h	precision	fwd_time(usec)	fwd_algo	Ops (mill)
# 14	14	512	1	512	3	3	1	1	1	1	float	186	WINOGRAD	925
