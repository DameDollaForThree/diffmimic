from .HUMANOID import _SYSTEM_CONFIG_HUMANOID
from .SWORDSHIELD import _SYSTEM_CONFIG_SWORDSHIELD
from .SMPL import _SYSTEM_CONFIG_SMPL
from .A1 import _SYSTEM_CONFIG_A1

from google.protobuf import text_format
from brax.physics.config_pb2 import Config


def get_system_cfg(system_type):
    return {
      'humanoid': _SYSTEM_CONFIG_HUMANOID,
      'swordshield': _SYSTEM_CONFIG_SWORDSHIELD,
      'smpl': _SYSTEM_CONFIG_SMPL,
      'a1': _SYSTEM_CONFIG_A1
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())