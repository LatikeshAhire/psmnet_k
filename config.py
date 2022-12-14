# 数据集参数
IMG_WIDTH = 960
IMG_HEIGHT = 540
IMG_N_CHANNEL = 3

# 训练参数
TRAIN_BATCH_SIZE = 3
TRAIN_CROP_WIDTH = 512
TRAIN_CROP_HEIGHT = 256

# 学习率等参数
TRAIN_LR = 0.0001  # 0.002
TRAIN_LR_DECAY = 0.1
TRAIN_LR_DECAY_EPOCH = 30
TRAIN_EPOCH = 10
# 三个分支loss的比例
TRAIN_LOSS_COEF = (0.5, 0.7, 1.0)
# 3d cnn的类型
HEAD_STACKED_HOURGLASS = 0
HEAD_BASIC = 1
L2_REG = 1e-4
MAX_DISP = 192

VAL_BATCH_SIZE = 18

LOG_INTERVAL = 10

SCENEFLOW_SIZE = (540, 960)
KITTI2015_SIZE = (384, 1248)
