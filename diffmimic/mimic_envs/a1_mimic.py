import brax
from brax.v1 import jumpy as jp
import numpy as np
from brax.envs import base
from brax.io import mjcf
from diffmimic.utils.io import deserialize_qp
from .losses import *
from diffmimic.utils.rotation6d import quaternion_to_rotation_6d
import jax

class A1Mimic(base.PipelineEnv):
    """Trains an A1 to mimic reference motion."""

    def __init__(self, reference_traj, obs_type='timestamp', cyc_len=None, reward_scaling=1.,
                 pos_weight=1., rot_weight=1., vel_weight=0., ang_weight=0., n_frames=10, 
                 root_pos_xy_weight=0., root_pos_z_weight=0., root_ori_weight=0., joint_weight=0.):
        path = '/data/benny_cai/diffmimic/diffmimic/mimic_envs/system_configs'
        with open(path + '/a1_mjcf.xml', 'r') as file:
            config = file.read()
        self.sys = mjcf.loads(config, asset_path=path)
        # bias_qd = -10 * np.ones(12, dtype=np.float32) # add D term in PD controller
        # self.sys = self.sys.replace(actuator=self.sys.actuator.replace(bias_qd=bias_qd))
        backend = 'positional'
        super().__init__(sys=self.sys, backend=backend, n_frames=n_frames)
        self.reference_qp = deserialize_qp(reference_traj)
        self.reference_len = reference_traj.shape[0]
        self.reward_scaling = reward_scaling
        self.obs_type = obs_type
        self.cycle_len = cyc_len if cyc_len is not None else self.reference_len
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.vel_weight = vel_weight
        self.ang_weight = ang_weight
        # self.root_pos_xy_weight = root_pos_xy_weight
        # self.root_pos_z_weight = root_pos_z_weight
        # self.root_ori_weight = root_ori_weight
        # self.joint_weight = joint_weight

    def reset(self, rng: jp.ndarray) -> base.State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)
        qp = self._get_ref_state(zero) # retrieve the 1st ref state
        metrics = {'step_index': zero, 'pose_error': zero, 'fall': zero, 'thres_error': zero, 'mse_pos': zero, 'mse_rot': zero, 
                   'mse_vel': zero, 'mse_ang': zero, 'mse_root_pos_xy': zero, 'mse_root_pos_z': zero, 'mse_root_ori': zero, 'mse_joint': zero}
        obs = self._get_obs(qp, step_index=zero)
        state = base.State(qp, obs, reward, done, metrics)
        return state

    def step(self, state: base.State, action: jp.ndarray) -> base.State:
        """Run one timestep of the environment's dynamics."""
        step_index = state.metrics['step_index'] + 1
        
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        qp = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(qp, step_index)
        ref_qp = self._get_ref_state(step_idx=step_index) # retrieve the next ref state
        reward = -1 * (self.pos_weight * mse_pos(qp, ref_qp) +
                       self.rot_weight * mse_rot(qp, ref_qp) +
                       self.vel_weight * mse_vel(qp, ref_qp) +
                       self.ang_weight * mse_ang(qp, ref_qp)
                    #    + self.root_pos_xy_weight * mse_root_pos_xy(qp, ref_qp)
                    #    + self.root_pos_z_weight * mse_root_pos_z(qp, ref_qp)
                    #    + self.root_ori_weight * mse_root_ori(qp, ref_qp)
                    #    + self.joint_weight * mse_joint(qp, ref_qp)
                       ) * self.reward_scaling
        # TODO: fall: below 0.1 or above 1
        fall = jp.where(qp.x_i.pos[0, 2] < 0.1, jp.float32(1), jp.float32(0))
        fall = jp.where(qp.x_i.pos[0, 2] > 1, jp.float32(1), fall)
        state.metrics.update(
            step_index=step_index,
            pose_error=loss_l2_relpos(qp, ref_qp),
            fall=fall,
            mse_pos=mse_pos(qp, ref_qp),
            mse_rot=mse_rot(qp, ref_qp),
            mse_vel=mse_vel(qp, ref_qp),
            mse_ang=mse_ang(qp, ref_qp),
            mse_root_pos_xy=mse_root_pos_xy(qp, ref_qp),
            mse_root_pos_z=mse_root_pos_z(qp, ref_qp),
            mse_root_ori=mse_root_ori(qp, ref_qp),
            mse_joint=mse_joint(qp, ref_qp),
            thres_error=loss_l2_pos(qp, ref_qp)
        )
        state = state.replace(pipeline_state=qp, obs=obs, reward=reward)
        return state

    def _get_obs(self, qp: brax.positional.base.State, step_index: jp.ndarray) -> jp.ndarray:
        """Observe a1 body position, velocities, and angles."""
        pos, rot, vel, ang = qp.x_i.pos, qp.x_i.rot, qp.xd_i.vel, qp.xd_i.ang
        rot_6d = quaternion_to_rotation_6d(rot)
        rel_pos = (pos - pos[0])[1:]

        if self.obs_type == 'timestamp':
            phi = (step_index % self.cycle_len) / self.cycle_len
            obs = jp.concatenate([rel_pos.reshape(-1), rot_6d.reshape(-1), vel.reshape(-1), ang.reshape(-1),
                                  phi[None]], axis=-1)
        elif self.obs_type == 'target_state':
            target_qp = self._get_ref_state(step_idx=step_index + 1)
            target_pos, target_rot, target_vel, target_ang = target_qp.x_i.pos, target_qp.x_i.rot, target_qp.xd_i.vel, target_qp.xd_i.ang
            target_rot_6d = quaternion_to_rotation_6d(target_rot)
            obs = jp.concatenate([pos.reshape(-1), rot.reshape(-1), vel.reshape(-1), ang.reshape(-1),
                                  target_pos.reshape(-1), target_rot_6d.reshape(-1), target_vel.reshape(-1), target_ang.reshape(-1)
                                  ], axis=-1)
        # elif self.obs_type == 'joint':
        #     phi = (step_index % self.cycle_len) / self.cycle_len
        #     obs = jp.concatenate([rel_pos.reshape(-1), rot_6d.reshape(-1), vel.reshape(-1), ang.reshape(-1), qp.q, qp.qd,
        #                             phi[None]], axis=-1)
        else:
            raise NotImplementedError
        return obs

    def _get_ref_state(self, step_idx) -> brax.positional.base.State:
        mask = jp.where(step_idx == jp.arange(0, self.reference_len), jp.float32(1), jp.float32(0))
        q = mask @ self.reference_qp.q
        qd = mask @ self.reference_qp.qd
        x_pos = mask @ self.reference_qp.x_pos.transpose(1,0,2)
        x_rot = mask @ self.reference_qp.x_rot.transpose(1,0,2)
        xd_vel = mask @ self.reference_qp.xd_vel.transpose(1,0,2)
        xd_ang = mask @ self.reference_qp.xd_ang.transpose(1,0,2)
        xi_pos = mask @ self.reference_qp.xi_pos.transpose(1,0,2)
        xi_rot = mask @ self.reference_qp.xi_rot.transpose(1,0,2)
        xdi_vel = mask @ self.reference_qp.xdi_vel.transpose(1,0,2)
        xdi_ang = mask @ self.reference_qp.xdi_ang.transpose(1,0,2)
        j_pos = mask @ self.reference_qp.j_pos.transpose(1,0,2)
        j_rot = mask @ self.reference_qp.j_rot.transpose(1,0,2)
        jd_vel = mask @ self.reference_qp.jd_vel.transpose(1,0,2)
        jd_ang = mask @ self.reference_qp.jd_ang.transpose(1,0,2)
        ap_pos = mask @ self.reference_qp.ap_pos.transpose(1,0,2)
        ap_rot = mask @ self.reference_qp.ap_rot.transpose(1,0,2)
        ac_pos = mask @ self.reference_qp.ac_pos.transpose(1,0,2)
        ac_rot = mask @ self.reference_qp.ac_rot.transpose(1,0,2)
        mass = mask @ self.reference_qp.mass
        return brax.positional.base.State(q=q, qd=qd,
            x = brax.base.Transform(pos=x_pos, rot=x_rot),
            xd = brax.base.Motion(vel=xd_vel, ang=xd_ang), 
            contact=None,
            x_i = brax.base.Transform(pos=xi_pos, rot=xi_rot),
            xd_i = brax.base.Motion(vel=xdi_vel, ang=xdi_ang),
            j = brax.base.Transform(pos=j_pos, rot=j_rot),
            jd = brax.base.Motion(vel=jd_vel, ang=jd_ang),
            a_p = brax.base.Transform(pos=ap_pos, rot=ap_rot),
            a_c = brax.base.Transform(pos=ac_pos, rot=ac_rot),
            mass = mass
        )