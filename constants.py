import torch

SIZE = (128, 128)
LR = 3e-4
EPOCHS = 18
INPUT_PATH: str = "../stanford-car-dataset-by-classes-folder/car_data/train"
TEST_PATH: str = "../stanford-car-dataset-by-classes-folder/car_data/test"
BATCH_SIZE = 64
TBOARD_PATH: str = "runs"
LOAD_MODEL = ""
# LOAD_MODEL: str = "./saved_models/mnas_xs_19_10_2019_14_10/mnas_xs3.pth"
LOAD_BACKBONE = ""
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
