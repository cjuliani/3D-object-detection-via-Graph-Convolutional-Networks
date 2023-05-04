import tensorflow as tf
import random

# Dataset.
VOXEL_SIZE = lambda : random.choice([3])  # WARNING: the higher the size, the less accurate for small objects
POINTSET_RADIUS = lambda : random.choice([1])   # >=2 if points from cloud have 1m distance interval

# Model.
MAX_ANGLE = 90.     # for angle class convertion in <dataset>
ANGLE_BINS = 9
SEG_UNITS = [32,32,16,8,2]
VOTING_UNITS = [32, 32, 32+3]  # last number represent feature + 3 coordinates
PROPOSAL_UNITS = [32, 32, 32, 2 + (ANGLE_BINS * 2) + 3]  # semantics + (num_ang_bins*2) + size
GROUP_RADIUS = [10, 20, 30]

# Training.
MODEL_DIR = 'train_6' + '/'  # name of folder/experiment to save summaries and weights
MODEL_NAME = "model.ckpt"  # NOTE: No need to change.
MAX_ITER = 300000
SUMMARY_ITER = 1
VALIDATION_ITER = 5
SAVE_ITER = 3
LR = 1e-3
LR_CHANGE_EPOCH = 2
LR_CHANGE_VAL = 10
LR_MINIMUM = 1e-3
LOSS_WEIGHTS = [50., 1., 0.1, 0.1, 0.01, 0.001]    # sem, center, angle_cls, angle_res, sizes, corners
SEG_WEIGHT_1s, SEG_WEIGHT_0s = 0.75, 0.25  # segmentation weights (0 to 1), if 1 is minority class, weighted higher
NUMB_OBJ_TO_LEARN = 1    # number of best objects to learn from in batch
IOU_NUMBER = 80     # number of IOU calculations considered (random IOUs selected from batch) to avoid slow metric calc.
OBJ_RATIO = 0.75     # the ratio of object points from batch over the original number of points constituting the object
BATCH_WINDOW_RADIUS = lambda : random.choice([50])
DETECT_WINDOW_RADIUS = 100

BIAS = True
AUGMENT= False
IS_TRAINING = False
BATCH_NORM = False
WEIGHTS_REG = tf.keras.regularizers.L1L2(0.1, 0.0)

# Define data folders to use for training. Each folder
# consists of pre-processed point cloud data.
FOLDERS_TO_USE = ['NPD18P'+str(i) for i in [34,35,30,36,31,32,38,39,40,28,32,22,19,57,58]]

# --- Directories
DATA_ROOT = './dataset/'
SUMMARY_DIR = './summary/'
CHECKPOINT_DIR = './checkpoint/'
