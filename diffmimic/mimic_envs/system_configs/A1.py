_SYSTEM_CONFIG_A1 = """
bodies {
  name: "trunk"
  colliders {
    position {
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1335
        y: 0.097
        z: 0.057
      }
    }
  }
  colliders {
    position {
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.0005
        y: 0.0005
        z: 0.0005
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.7139997
}
bodies {
  name: "FR_hip"
  colliders {
    position {
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position {
      y: -0.081
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.69699997
}
bodies {
  name: "FR_upper"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.01225
        z: 0.017
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.013
}
bodies {
  name: "FR_lower"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.008
        z: 0.008
      }
    }
  }
  colliders {
    position {
      z: -0.2
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    sphere {
      radius: 0.02
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.226
}
bodies {
  name: "FL_hip"
  colliders {
    position {
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position {
      y: 0.081
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.69699997
}
bodies {
  name: "FL_upper"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.01225
        z: 0.017
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.013
}
bodies {
  name: "FL_lower"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.008
        z: 0.008
      }
    }
  }
  colliders {
    position {
      z: -0.2
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    sphere {
      radius: 0.02
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.226
}
bodies {
  name: "RR_hip"
  colliders {
    position {
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position {
      y: -0.081
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.69699997
}
bodies {
  name: "RR_upper"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.01225
        z: 0.017
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.013
}
bodies {
  name: "RR_lower"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.008
        z: 0.008
      }
    }
  }
  colliders {
    position {
      z: -0.2
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    sphere {
      radius: 0.02
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.226
}
bodies {
  name: "RL_hip"
  colliders {
    position {
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.092
    }
  }
  colliders {
    position {
      y: 0.081
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.041
      length: 0.102
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.69699997
}
bodies {
  name: "RL_upper"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.01225
        z: 0.017
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.013
}
bodies {
  name: "RL_lower"
  colliders {
    position {
      z: -0.1
    }
    rotation {
      x: -0.0
      y: 90.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.1
        y: 0.008
        z: 0.008
      }
    }
  }
  colliders {
    position {
      z: -0.2
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    sphere {
      radius: 0.02
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
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
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "FR_hip"
  parent: "trunk"
  child: "FR_hip"
  parent_offset {
    x: 0.183
    y: -0.047
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -46.0
    max: 46.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "FR_upper"
  parent: "FR_hip"
  child: "FR_upper"
  parent_offset {
    y: -0.08505
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -60.0
    max: 180.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "FR_lower"
  parent: "FR_upper"
  child: "FR_lower"
  parent_offset {
    z: -0.2
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -154.5
    max: -52.5
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "FL_hip"
  parent: "trunk"
  child: "FL_hip"
  parent_offset {
    x: 0.183
    y: 0.047
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -46.0
    max: 46.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "FL_upper"
  parent: "FL_hip"
  child: "FL_upper"
  parent_offset {
    y: 0.08505
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -60.0
    max: 180.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "FL_lower"
  parent: "FL_upper"
  child: "FL_lower"
  parent_offset {
    z: -0.2
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -154.5
    max: -52.5
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RR_hip"
  parent: "trunk"
  child: "RR_hip"
  parent_offset {
    x: -0.183
    y: -0.047
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -46.0
    max: 46.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RR_upper"
  parent: "RR_hip"
  child: "RR_upper"
  parent_offset {
    y: -0.08505
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -60.0
    max: 180.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RR_lower"
  parent: "RR_upper"
  child: "RR_lower"
  parent_offset {
    z: -0.2
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -154.5
    max: -52.5
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RL_hip"
  parent: "trunk"
  child: "RL_hip"
  parent_offset {
    x: -0.183
    y: 0.047
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -46.0
    max: 46.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RL_upper"
  parent: "RL_hip"
  child: "RL_upper"
  parent_offset {
    y: 0.08505
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -60.0
    max: 180.0
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "RL_lower"
  parent: "RL_upper"
  child: "RL_lower"
  parent_offset {
    z: -0.2
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -154.5
    max: -52.5
  }
  spring_damping: 50.0
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
actuators {
  name: "FR_hip"
  joint: "FR_hip"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "FR_upper"
  joint: "FR_upper"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "FR_lower"
  joint: "FR_lower"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "FL_hip"
  joint: "FL_hip"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "FL_upper"
  joint: "FL_upper"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "FL_lower"
  joint: "FL_lower"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RR_hip"
  joint: "RR_hip"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RR_upper"
  joint: "RR_upper"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RR_lower"
  joint: "RR_lower"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RL_hip"
  joint: "RL_hip"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RL_upper"
  joint: "RL_upper"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "RL_lower"
  joint: "RL_lower"
  strength: 100.0
  torque {
  }
}
friction: 0.6
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
}
dt: 0.02
substeps: 4
"""