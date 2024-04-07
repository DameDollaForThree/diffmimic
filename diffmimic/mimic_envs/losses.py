from diffmimic.utils.rotation6d import quaternion_to_rotation_6d


def loss_l2_relpos(qp, ref_qp):
    pos, ref_pos = qp.x_i.pos, ref_qp.x_i.pos
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
    relpos_loss = relpos_loss
    return relpos_loss


def loss_l2_pos(qp, ref_qp):
    pos, ref_pos = qp.x_i.pos, ref_qp.x_i.pos
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss


def mse_pos(qp, ref_qp):
    pos, ref_pos = qp.x_i.pos, ref_qp.x_i.pos
    pos_loss = ((pos - ref_pos) ** 2).sum(-1).mean()
    return pos_loss


def mse_rot(qp, ref_qp):
    rot, ref_rot = quaternion_to_rotation_6d(qp.x_i.rot), quaternion_to_rotation_6d(ref_qp.x_i.rot)
    rot_loss = ((rot - ref_rot) ** 2).sum(-1).mean()
    return rot_loss


def mse_vel(qp, ref_qp):
    vel, ref_vel = qp.xd_i.vel, ref_qp.xd_i.vel
    vel_loss = ((vel - ref_vel) ** 2).sum(-1).mean()
    return vel_loss


def mse_ang(qp, ref_qp, reduce='mean'):
    ang, ref_ang = qp.xd_i.ang, ref_qp.xd_i.ang
    ang_loss = ((ang - ref_ang) ** 2).sum(-1).mean()
    return ang_loss

# added by me
def mse_root_pos_xy(qp, ref_qp):
    root_pos_xy, ref_root_pos_xy = qp.q[:2], ref_qp.q[:2]
    root_pos_xy_loss = ((root_pos_xy - ref_root_pos_xy) ** 2).mean()
    return root_pos_xy_loss

def mse_root_pos_z(qp, ref_qp):
    root_pos_z, ref_root_pos_z = qp.q[2], ref_qp.q[2]
    root_pos_z_loss = ((root_pos_z - ref_root_pos_z) ** 2).mean()
    return root_pos_z_loss

def mse_root_ori(qp, ref_qp):
    root_ori, ref_root_ori = quaternion_to_rotation_6d(qp.q[3:7]), quaternion_to_rotation_6d(ref_qp.q[3:7])
    root_ori_loss = ((root_ori - ref_root_ori) ** 2).mean()
    return root_ori_loss

def mse_joint(qp, ref_qp):
    joint, ref_joint = qp.q[7:], ref_qp.q[7:]
    joint_loss = ((joint - ref_joint) ** 2).mean()
    return joint_loss