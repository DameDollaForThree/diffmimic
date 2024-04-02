import numpy as np
import json
import jax
from jax import numpy as jp
from brax.positional import pipeline
from brax.io import mjcf
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


if __name__ == "__main__":
    motion = "jump08"
    file = f'a1_origin_motions/{motion}.txt'
    with open(file, 'r') as f:
        data = json.load(f)
    frames = data['Frames']
    frames = np.array(frames)
    frameDuration = data['FrameDuration']
    loopMode = data['LoopMode']
    print("Motion: ", motion)
    print("Original shape: ", frames.shape)
    
    targetDuration = 0.05
    interpolated_idx = 0
    interp_frames = []
    for i in range(frames.shape[0]-1):
        if interpolated_idx * targetDuration >= frameDuration * i \
            and interpolated_idx * targetDuration <= frameDuration * (i+1):
            frame0, frame1 = frames[i], frames[i+1]
            
            blend = (interpolated_idx * targetDuration - frameDuration * i) / frameDuration

            root_pos0 = frame0[0:3]
            root_pos1 = frame1[0:3]
            root_rot0 = frame0[3:7]
            root_rot1 = frame1[3:7]
            joints0 = frame0[7:19]
            joints1 = frame1[7:19]

            quats = [root_rot0, root_rot1]
            quats = [[quat[3], quat[0], quat[1], quat[2]] for quat in quats]
            key_rots = R.from_quat(quats)
            key_times = [frameDuration * i, frameDuration * (i+1)]
            slerp = Slerp(key_times, key_rots)
            time = interpolated_idx * targetDuration
            interp_rots = slerp(time)
            
            blend_root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
            blend_joints = (1.0 - blend) * joints0 + blend * joints1
            interp_frames.append(np.hstack((blend_root_pos, interp_rots.as_quat(), blend_joints)))
            interpolated_idx += 1
    interp_frames = np.array(interp_frames)
    print("Interpolated shape: ", interp_frames.shape)
    
    with open('a1_mjcf.txt', 'r') as file:
        config = file.read()

    m = mjcf.loads(config, asset_path='')
        
    jit_env_reset = jax.jit(pipeline.init)
    jit_env_step = jax.jit(pipeline.step)

    rng = jax.random.PRNGKey(seed=1)
    rollout = []
    for frame in interp_frames:
        state = jit_env_reset(m, frame, jp.zeros(m.qd_size()))
        rollout.append(state)
    
    # serialize data, positional:
    data = []
    for roll in rollout:
        QP = np.hstack((
            roll.q, 
            roll.qd, 
            roll.x.pos.flatten(), 
            roll.x.rot.flatten(), 
            roll.xd.vel.flatten(), 
            roll.xd.ang.flatten(),
            roll.x_i.pos.flatten(), 
            roll.x_i.rot.flatten(), 
            roll.xd_i.vel.flatten(), 
            roll.xd_i.ang.flatten(),
            roll.j.pos.flatten(), 
            roll.j.rot.flatten(), 
            roll.jd.vel.flatten(), 
            roll.jd.ang.flatten(),
            roll.a_p.pos.flatten(), 
            roll.a_p.rot.flatten(), 
            roll.a_c.pos.flatten(), 
            roll.a_c.rot.flatten(),
            roll.mass
        ))
        data.append(QP)
        
    data = np.array(data)
    print("data shape: ", data.shape)
    np.save(f'a1_ref_motion/{motion}_{targetDuration}.npy', data)