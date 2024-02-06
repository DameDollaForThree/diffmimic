_SYSTEM_CONFIG_A1 = """
bodies {
  name: "trunk"
  colliders {
    box {
      halfsize { x: 0.1335 y: 0.097 z: 0.057 }
    }
  }
  inertia { x: 0.0158533 y: 0.0377999 z: 0.0456542 }
  mass: 4.7139997
}
bodies {
  name: "FR_hip"
  colliders {
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position { y: -0.081 }
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia { x: 0.000807752 y: 0.00055293 z: 0.000468983 }
  mass: 0.69699997
}
bodies {
  name: "FR_upper"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.01225 z: 0.017 }
    }
  }
  inertia { x: 0.00555739 y: 0.00513936 z: 0.00133944 }
  mass: 1.013
}
bodies {
  name: "FR_lower"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.008 z: 0.008 }
    }
  }
  colliders {
    position { z: -0.2 }
    sphere { radius: 0.02 }
  }
  inertia { x: 0.00340344 y: 0.00339393 z: 0.0034 }
  mass: 0.226
}
bodies {
  name: "FL_hip"
  colliders {
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position { y: 0.081 }
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia { x: 0.000807752 y: 0.00055293 z: 0.000468983 }
  mass: 0.69699997
}
bodies {
  name: "FL_upper"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.01225 z: 0.017 }
    }
  }
  inertia { x: 0.00555739 y: 0.00513936 z: 0.00133944 }
  mass: 1.013
}
bodies {
  name: "FL_lower"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.008 z: 0.008 }
    }
  }
  colliders {
    position { z: -0.2 }
    sphere { radius: 0.02 }
  }
  inertia { x: 0.00340344 y: 0.00339393 z: 0.0034 }
  mass: 0.226
}
bodies {
  name: "RR_hip"
  colliders {
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position { y: -0.081 }
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia { x: 0.000807752 y: 0.00055293 z: 0.000468983 }
  mass: 0.69699997
}
bodies {
  name: "RR_upper"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.01225 z: 0.017 }
    }
  }
  inertia { x: 0.00555739 y: 0.00513936 z: 0.00133944 }
  mass: 1.013
}
bodies {
  name: "RR_lower"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.008 z: 0.008 }
    }
  }
  colliders {
    position { z: -0.2 }
    sphere { radius: 0.02 }
  }
  inertia { x: 0.00340344 y: 0.00339393 z: 0.0034 }
  mass: 0.226
}
bodies {
  name: "RL_hip"
  colliders {
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position { y: 0.081 }
    rotation { x: 90.0 z: -0.0 }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia { x: 0.000807752 y: 0.00055293 z: 0.000468983 }
  mass: 0.69699997
}
bodies {
  name: "RL_upper"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.01225 z: 0.017 }
    }
  }
  inertia { x: 0.00555739 y: 0.00513936 z: 0.00133944 }
  mass: 1.013
}
bodies {
  name: "RL_lower"
  colliders {
    position { z: -0.1 }
    rotation { y: 90.0 }
    box {
      halfsize { x: 0.1 y: 0.008 z: 0.008 }
    }
  }
  colliders {
    position { z: -0.2 }
    sphere { radius: 0.02 }
  }
  inertia { x: 0.00340344 y: 0.00339393 z: 0.0034 }
  mass: 0.226
}
bodies {
    name: "floor"
    colliders {
      plane {
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    frozen {
      position { x: 1.0 y: 1.0 z: 1.0 }
      rotation { x: 1.0 y: 1.0 z: 1.0 }
    }
  }
joints {
  name: "trunk_FR_hip"
  angle_limit { min: -46 max: 46 }
  parent_offset { x: 0.183 y: -0.047 }
  parent: "trunk"
  child: "FR_hip"
  angular_damping: 0.0
}
joints {
  name: "FR_hip_FR_upper"
  angle_limit { min: -60 max: 180 }
  rotation { z: 90 }
  parent_offset { y: -0.08505 }
  parent: "FR_hip"
  child: "FR_upper"
  angular_damping: 0.0
}
joints {
  name: "FR_upper_FR_lower"
  angle_limit { min: -154.5 max: -52.5 }
  rotation { z: 90 }
  parent_offset { z: -0.2 }
  parent: "FR_upper"
  child: "FR_lower"
  angular_damping: 0.0
}
joints {
  name: "trunk_FL_hip"
  angle_limit { min: -46 max: 46 }
  parent_offset { x: 0.183 y: 0.047 }
  parent: "trunk"
  child: "FL_hip"
  angular_damping: 0.0
}
joints {
  name: "FL_hip_FL_upper"
  angle_limit { min: -60 max: 180 }
  rotation { z: 90 }
  parent_offset { y: 0.08505 }
  parent: "FL_hip"
  child: "FL_upper"
  angular_damping: 0.0
}
joints {
  name: "FL_upper_FL_lower"
  angle_limit { min: -154.5 max: -52.5 }
  rotation { z: 90 }
  parent_offset { z: -0.2 }
  parent: "FL_upper"
  child: "FL_lower"
  angular_damping: 0.0
}
joints {
  name: "trunk_RR_hip"
  angle_limit { min: -46 max: 46 }
  parent_offset { x: -0.183 y: -0.047 }
  parent: "trunk"
  child: "RR_hip"
  angular_damping: 0.0
}
joints {
  name: "RR_hip_RR_upper"
  angle_limit { min: -60 max: 180 }
  rotation { z: 90 }
  parent_offset { y: -0.08505 }
  parent: "RR_hip"
  child: "RR_upper"
  angular_damping: 0.0
}
joints {
  name: "RR_upper_RR_lower"
  angle_limit { min: -154.5 max: -52.5 }
  rotation { z: 90 }
  parent_offset {z: -0.2 }
  parent: "RR_upper"
  child: "RR_lower"
  angular_damping: 0.0
}
joints {
  name: "trunk_RL_hip"
  angle_limit { min: -46 max: 46 }
  parent_offset { x: -0.183 y: 0.047 }
  parent: "trunk"
  child: "RL_hip"
  angular_damping: 0.0
}
joints {
  name: "RL_hip_RL_upper"
  angle_limit { min: -60 max: 180 }
  rotation { z: 90 }
  parent_offset { y: 0.08505 }
  parent: "RL_hip"
  child: "RL_upper"
  angular_damping: 0.0
}
joints {
  name: "RL_upper_RL_lower"
  angle_limit { min: -154.5 max: -52.5 }
  rotation { z: 90 }
  parent_offset { z: -0.2 }
  parent: "RL_upper"
  child: "RL_lower"
  angular_damping: 0.0
}
actuators {
  name: "trunk_FR_hip"
  angle {}
  joint: "trunk_FR_hip"
  strength: 50.0
}
actuators {
  name: "FR_hip_FR_upper"
  angle {}
  joint: "FR_hip_FR_upper"
  strength: 50.0
}
actuators {
  name: "FR_upper_FR_lower"
  angle {}
  joint: "FR_upper_FR_lower"
  strength: 50.0
}
actuators {
  name: "trunk_FL_hip"
  angle {}
  joint: "trunk_FL_hip"
  strength: 50.0
}
actuators {
  name: "FL_hip_FL_upper"
  angle {}
  joint: "FL_hip_FL_upper"
  strength: 50.0
}
actuators {
  name: "FL_upper_FL_lower"
  angle {}
  joint: "FL_upper_FL_lower"
  strength: 50.0
}
actuators {
  name: "trunk_RR_hip"
  angle {}
  joint: "trunk_RR_hip"
  strength: 50.0
}
actuators {
  name: "RR_hip_RR_upper"
  angle {}
  joint: "RR_hip_RR_upper"
  strength: 50.0
}
actuators {
  name: "RR_upper_RR_lower"
  angle {}
  joint: "RR_upper_RR_lower"
  strength: 50.0
}
actuators {
  name: "trunk_RL_hip"
  angle {}
  joint: "trunk_RL_hip"
  strength: 50.0
}
actuators {
  name: "RL_hip_RL_upper"
  angle {}
  joint: "RL_hip_RL_upper"
  strength: 50.0
}
actuators {
  name: "RL_upper_RL_lower"
  angle {}
  joint: "RL_upper_RL_lower"
  strength: 50.0
}
friction: 0.77459666924
gravity { z: -9.81 }
angular_damping: -0.009999999776482582
collide_include {
  first: "floor"
  second: "trunk"
}
collide_include {
  first: "floor"
  second: "FR_hip"
}
collide_include {
  first: "floor"
  second: "FR_upper"
}
collide_include {
  first: "floor"
  second: "FR_lower"
}
collide_include {
  first: "floor"
  second: "FL_hip"
}
collide_include {
  first: "floor"
  second: "FL_upper"
}
collide_include {
  first: "floor"
  second: "FL_lower"
}
collide_include {
  first: "floor"
  second: "RR_hip"
}
collide_include {
  first: "floor"
  second: "RR_upper"
}
collide_include {
  first: "floor"
  second: "RR_lower"
}
collide_include {
  first: "floor"
  second: "RL_hip"
}
collide_include {
  first: "floor"
  second: "RL_upper"
}
collide_include {
  first: "floor"
  second: "RL_lower"
}
frozen {
  position {
    y: 1.0
  }
  rotation {
    x: 1.0
    z: 1.0
  }
}
dt: 0.05
substeps: 16
dynamics_mode: "pbd"
"""