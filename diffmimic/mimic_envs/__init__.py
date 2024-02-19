from brax import envs

from .a1_mimic import A1Mimic
from .a1_mimic_train import A1MimicTrain
from . import pd_controller


def register_mimic_env():
    envs.register_environment('a1_mimic', A1Mimic)
    envs.register_environment('a1_mimic_train', A1MimicTrain)
