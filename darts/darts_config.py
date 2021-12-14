# Constants for DARTS training configuration. These should be constant for a
# complete set of experiments.

# TRAIN
BATCH_SIZE = 256
LR = 0.025
MOMENTUM = 0.9
WD = 3e-4
INIT_CHANNELS = 40
LAYERS = 4
GRAD_CLIP = 5
DROPPATH_PROB = 0.1 # Probability of dropping a path
CUTOUT_LENGTH = 16
AUXILIARY_WEIGHT = 0.4
REPORT_FREQ = 50

## EVAL
EVAL_BATCH_SIZE = 96
EVAL_INIT_CHANNEL = 36
EVAL_LAYERS = 20

