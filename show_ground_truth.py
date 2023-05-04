import tensorflow as tf
import argparse

from src.solver import Solver

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str)
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()

# Define solver object.
solver = Solver()


if __name__ == '__main__':
    solver.show_ground_truth(folder=ARGS.data_folder)
