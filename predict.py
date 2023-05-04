import tensorflow as tf
import argparse

from src.solver import Solver

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.1, help='Memory fraction of GPU used for training.')
parser.add_argument("--gpu_allow_growth", default=False)
parser.add_argument("--gpu_device", default='0')
parser.add_argument("--soft_placement", default=True)
parser.add_argument("--model_folder", type=str, default='train_1')
parser.add_argument("--data_folder", type=str)
parser.add_argument("--radius", default=10)
parser.add_argument("--NMS", default=False)
parser.add_argument("--overlap_ratio", default=0.1)
parser.add_argument("--iou_thresh", default=0.5)
parser.add_argument("--semantics_only", default=False)
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()

# Define solver object.
solver = Solver()

tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=bool(ARGS.soft_placement))
tf_config.gpu_options.allow_growth = bool(ARGS.gpu_allow_growth)
tf_config.gpu_options.per_process_gpu_memory_fraction = float(ARGS.gpu_memory)
tf_config.gpu_options.visible_device_list = str(ARGS.gpu_device)


if __name__ == '__main__':
    solver.detect(
        model=ARGS.model,
        sess_config=tf_config,
        folder=ARGS.data_folder,
        cluster_radius=float(ARGS.radius),
        NMS=bool(ARGS.NMS),
        overlap_ratio=float(ARGS.overlap_ratio),
        iou_thresh=float(ARGS.detect_iou_thresh),
        semantics_only=bool(ARGS.semantics_only))
