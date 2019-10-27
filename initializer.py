import torch
from models.mnas_classifier import MnasXs
from models.mnas_detection import MnasExtra
from constants import LOAD_MODEL, DEVICE, SIZE


def load_model(model_name, input_size=SIZE[0]):
    if model_name == 'mnas_xs':
        net = MnasXs(input_size=input_size).to(DEVICE)
    if model_name == 'mnas_extra':
        net = MnasExtra(input_size=input_size, previous_input_size=196).to(DEVICE)
    if LOAD_MODEL:
        print("Loading model from {}...".format(LOAD_MODEL))
        net.load_state_dict(torch.load(LOAD_MODEL))

    return net
