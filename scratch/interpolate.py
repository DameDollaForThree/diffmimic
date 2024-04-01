import numpy as np
import json
import jax
from jax import numpy as jp
from brax.positional import pipeline
from brax.io import mjcf

def standardize_quaternion(quat):
    mask = quat[..., -1] < 0
    quat[mask] = -quat[mask]
    return quat

def quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True, eps=1e-6):
    if isinstance(fraction, (int, float)):
        fraction = np.array([fraction])
    batch_size = quat0.shape[0]
    q = np.zeros_like(quat0)
    q0 = quat0[..., :4] / \
        np.linalg.norm(quat0[..., :4], axis=-1).reshape([batch_size, 1])
    q1 = quat1[..., :4] / \
        np.linalg.norm(quat1[..., :4], axis=-1).reshape([batch_size, 1])
    fraction_zero = np.squeeze(fraction == 0.0)
    fraction_one = np.squeeze(fraction == 1.0)
    mask = np.logical_and(
        np.logical_not(fraction_zero),
        np.logical_not(fraction_one)
    )
    q[fraction_zero] = q0[fraction_zero]
    q[fraction_one] = q1[fraction_one]
    d = np.squeeze(np.matmul(q0.reshape([-1, 1, q0.shape[1]]),
                                q1.reshape([-1, q1.shape[1], 1])))
    d[np.logical_not(mask)] = 0.0  # set to dummy value
    q[np.logical_and(mask, abs(abs(d) - 1.0) < eps)
    ] = q0[np.logical_and(mask, abs(abs(d) - 1.0) < eps)]
    mask = np.logical_and(
        mask,
        abs(abs(d) - 1.0) >= eps
    )
    d = d[mask]
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[mask][d < 0.0] *= -1.0
    d = d.reshape([-1, 1])
    angle = np.arccos(d) + spin * np.pi
    isin = 1.0 / np.sin(angle)
    
    q0[mask] *= np.sin((1.0 - fraction[mask]) * angle) * isin
    q1[mask] *= np.sin(fraction[mask] * angle) * isin
    q0[mask] += q1[mask]
    q[mask] = q0[mask]
    return q



if __name__ == "__main__":
    motion = "trot"
    file = f'a1_origin_motions/{motion}.txt'
    with open(file, 'r') as f:
        data = json.load(f)
    frames = data['Frames']
    frames = np.array(frames)
    frameDuration = data['FrameDuration']
    loopMode = data['LoopMode']
    print("Original shape: ", frames.shape)
    
    targetDuration = 0.05
    interpolated_idx = 1
    interpolated_frames = [frames[0]]
    for i in range(frames.shape[0]-1):
        if interpolated_idx * targetDuration >= frameDuration * i \
            and interpolated_idx * targetDuration <= frameDuration * (i+1):
            frame0, frame1 = frames[i], frames[i+1]
            
            blend = (interpolated_idx * targetDuration - frameDuration * i) / frameDuration
            eps = 1e-2
            if abs(blend) <= eps:
                interpolated_frames.append(frame0)
            elif abs(blend - 1) <= eps:
                interpolated_frames.append(frame1)
            else:
                root_pos0 = frame0[0:3]
                root_pos1 = frame1[0:3]
                root_rot0 = frame0[3:7]
                root_rot1 = frame1[3:7]
                joints0 = frame0[7:19]
                joints1 = frame1[7:19]
                _root_rot0 = root_rot0[np.newaxis, ...]
                _root_rot1 = root_rot1[np.newaxis, ...]
                
                blend_root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
                blend_joints = (1.0 - blend) * joints0 + blend * joints1
                blend_root_rot = quat_slerp(_root_rot0, _root_rot1, blend)
                blend_root_rot /= np.linalg.norm(blend_root_rot, axis=-1).reshape([blend_root_rot.shape[0], 1])
                blend_root_rot = standardize_quaternion(blend_root_rot)[0]
                blend_root_rot = np.round(blend_root_rot, decimals=5)
                
                interpolated_frames.append(np.hstack((blend_root_pos, blend_root_rot, blend_joints)))
            interpolated_idx += 1
     
    interpolated_frames = np.array(interpolated_frames)
    print("Interpolated shape: ", interpolated_frames.shape)
    
    with open('a1_mjcf.txt', 'r') as file:
        config = file.read()

    m = mjcf.loads(config, asset_path='')
        
    jit_env_reset = jax.jit(pipeline.init)
    jit_env_step = jax.jit(pipeline.step)

    rng = jax.random.PRNGKey(seed=1)
    rollout = []
    for frame in interpolated_frames:
        frame[3], frame[4], frame[5], frame[6] = frame[6], frame[3], frame[4], frame[5]
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