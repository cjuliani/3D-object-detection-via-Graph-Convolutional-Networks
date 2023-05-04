import tensorflow as tf
import argparse

from src.solver import Solver

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.25, help='Memory fraction of GPU used for training.')
parser.add_argument("--gpu_allow_growth", default=False)
parser.add_argument("--gpu_device", default='0')
parser.add_argument("--soft_placement", default=True)
parser.add_argument("--save_model", default=False)
parser.add_argument("--model_folder", type=str, default='train_1')
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()

# Define solver object.
solver = Solver()

tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=bool(ARGS.soft_placement))
tf_config.gpu_options.allow_growth = bool(ARGS.gpu_allow_growth)
tf_config.gpu_options.per_process_gpu_memory_fraction = float(ARGS.gpu_memory)
tf_config.gpu_options.visible_device_list = str(ARGS.gpu_device)


if __name__ == '__main__':
    solver.train(
        sess_config=tf_config,
        save_model=bool(ARGS.save_model),
        model_to_restore=ARGS.model_folder)
