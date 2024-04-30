import numpy as np
import json
import jax
from jax import numpy as jp
from brax.positional import pipeline
from brax.io import mjcf
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import transforms3d.quaternions as quat


if __name__ == "__main__":
    motion = "a1_retarget_motion_left_turn0"
    file = f'a1_origin_motions/{motion}.txt'
    with open(file, 'r') as f:
        data = json.load(f)
    frames = data['Frames']
    frames = np.array(frames)
    frameDuration = data['FrameDuration']
    loopMode = data['LoopMode']
    print("Motion: ", motion)
    print("Original shape: ", frames.shape)
    
    # compute frame velocities
    num_frames = frames.shape[0]
    dt = frameDuration
    frames_vels = []

    for i in range(frames.shape[0]-1):
        frame0, frame1 = frames[i], frames[i+1]

        root_pos0 = frame0[0:3]
        root_pos1 = frame1[0:3]
        root_rot0 = frame0[3:7]
        root_rot1 = frame1[3:7]
        joints0 = frame0[7:19]
        joints1 = frame1[7:19]
        root_rot0 = [root_rot0[3], root_rot0[0], root_rot0[1], root_rot0[2]]
        root_rot1 = [root_rot1[3], root_rot1[0], root_rot1[1], root_rot1[2]]
        
        root_vel = (root_pos1 - root_pos0) / dt
        root_rot_diff = quat.qmult(root_rot1, quat.qconjugate(root_rot0))
        root_rot_diff_axis, root_rot_diff_angle = quat.quat2axangle(root_rot_diff)
        root_ang_vel = (root_rot_diff_angle / dt) * root_rot_diff_axis
        joints_vel = (joints1 - joints0) / dt
        curr_frame_vel = np.concatenate((root_vel, root_ang_vel, joints_vel))
        frames_vels.append(curr_frame_vel)

    # replicate the velocity at the last frame
    if num_frames > 1:
        frames_vels.append(frames_vels[-1])

    frames_vels = np.array(frames_vels)
    print("Original frame velocity shape", frames_vels.shape)
    
    targetDuration = 0.02
    print("targetDuration: ", targetDuration)
    
    if targetDuration == frameDuration:
        interp_frames = frames
        interp_frames_vels = frames_vels
        for i in range(interp_frames.shape[0]): 
            interp_frames[i,3], interp_frames[i,4], interp_frames[i,5], interp_frames[i,6] = \
                interp_frames[i,6], interp_frames[i,3], interp_frames[i,4], interp_frames[i,5]
    else:
        # Interpolation
        interpolated_idx = 0
        interp_frames = []
        interp_frames_vels = []
        for i in range(frames.shape[0]-1):
            if interpolated_idx * targetDuration >= frameDuration * i \
                and interpolated_idx * targetDuration <= frameDuration * (i+1):
                frame0, frame1 = frames[i], frames[i+1]
                frame_vel0, frame_vel1 = frames_vels[i], frames_vels[i+1]
                
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
                
                blend_frame_vel = (1.0 - blend) * frame_vel0 + blend * frame_vel1
                interp_frames_vels.append(blend_frame_vel)
                interpolated_idx += 1

        interp_frames = np.array(interp_frames)
        interp_frames_vels = np.array(interp_frames_vels)
         
    print("Interpolated frame shape: ", interp_frames.shape)
    print("Interpolated frame shape: ", interp_frames_vels.shape)
    
    # rollout
    with open('a1_mjcf.txt', 'r') as file:
        config = file.read()

    m = mjcf.loads(config, asset_path='')
        
    jit_env_reset = jax.jit(pipeline.init)
    jit_env_step = jax.jit(pipeline.step)

    rng = jax.random.PRNGKey(seed=1)
    rollout = []
    for frame, frame_vel in zip(interp_frames, interp_frames_vels):
        state = jit_env_reset(m, frame, frame_vel)
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
    print("save to: ", f'a1_ref_motion/{motion}_{int(1/targetDuration)}Hz.npy')
    np.save(f'a1_ref_motion/{motion}_{int(1/targetDuration)}Hz.npy', data)