import brax
import jax.numpy as jnp
import numpy as np
from diffmimic.utils.QP import QP


def deserialize_qp(nparray) -> QP:
    """
    Get QP from a trajectory numpy array
    """
    batch_dims = nparray.shape[:-1]
    slices = np.array([13 * x for x in [0, 3, 7, 10, 13, 16, 20, 23, 26, 29, 33, 36, 40, 41]])
    slices += 206
    slices = np.hstack(([0, 19, 37, 76, 128, 167], slices))
    q = jnp.reshape(nparray[..., slices[0]:slices[1]], batch_dims + (19, ))
    qd = jnp.reshape(nparray[..., slices[1]:slices[2]], batch_dims + (18, ))
    x_pos = jnp.reshape(nparray[..., slices[2]:slices[3]], batch_dims + (13, 3))
    x_rot = jnp.reshape(nparray[..., slices[3]:slices[4]], batch_dims + (13, 4))
    xd_vel = jnp.reshape(nparray[..., slices[4]:slices[5]], batch_dims + (13, 3))
    xd_ang = jnp.reshape(nparray[..., slices[5]:slices[6]], batch_dims + (13, 3))
    xi_pos = jnp.reshape(nparray[..., slices[6]:slices[7]], batch_dims + (13, 3))
    xi_rot = jnp.reshape(nparray[..., slices[7]:slices[8]], batch_dims + (13, 4))
    xdi_vel = jnp.reshape(nparray[..., slices[8]:slices[9]], batch_dims + (13, 3))
    xdi_ang = jnp.reshape(nparray[..., slices[9]:slices[10]], batch_dims + (13, 3))
    j_pos = jnp.reshape(nparray[..., slices[10]:slices[11]], batch_dims + (13, 3))
    j_rot = jnp.reshape(nparray[..., slices[11]:slices[12]], batch_dims + (13, 4))
    jd_vel = jnp.reshape(nparray[..., slices[12]:slices[13]], batch_dims + (13, 3))
    jd_ang = jnp.reshape(nparray[..., slices[13]:slices[14]], batch_dims + (13, 3))
    ap_pos = jnp.reshape(nparray[..., slices[14]:slices[15]], batch_dims + (13, 3))
    ap_rot = jnp.reshape(nparray[..., slices[15]:slices[16]], batch_dims + (13, 4))
    ac_pos = jnp.reshape(nparray[..., slices[16]:slices[17]], batch_dims + (13, 3))
    ac_rot = jnp.reshape(nparray[..., slices[17]:slices[18]], batch_dims + (13, 4))
    mass = jnp.reshape(nparray[..., slices[18]:slices[19]], batch_dims + (13, ))
    if batch_dims == ():
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
    return QP(q=q, qd=qd, x_pos=x_pos, x_rot=x_rot, xd_vel=xd_vel, xd_ang=xd_ang,
        xi_pos=xi_pos, xi_rot=xi_rot, xdi_vel=xdi_vel, xdi_ang=xdi_ang,
        j_pos=j_pos, j_rot=j_rot, jd_vel=jd_vel, jd_ang=jd_ang,
        ap_pos=ap_pos, ap_rot=ap_rot, ac_pos=ac_pos, ac_rot=ac_rot,
        mass=mass)


def serialize_qp(qp) -> jnp.array:
    """
    Serialize QP to a trajectory numpy array
    """
    q = qp.q
    qd = qp.qd
    x_pos = qp.x.pos
    x_rot = qp.x.rot
    xd_vel = qp.xd.vel
    xd_ang = qp.xd.ang
    xi_pos = qp.x_i.pos
    xi_rot = qp.x_i.rot
    xdi_vel = qp.xd_i.vel
    xdi_ang = qp.xd_i.ang
    j_pos = qp.j.pos
    j_rot = qp.j.rot
    jd_vel = qp.jd.vel
    jd_ang = qp.jd.ang
    ap_pos = qp.a_p.pos
    ap_rot = qp.a_p.rot
    ac_pos = qp.a_c.pos
    ac_rot = qp.a_c.rot
    mass = qp.mass
    batch_dim = x_pos.shape[:-2]
    nparray = []
    nparray.append(q.reshape(batch_dim + (-1,)))
    nparray.append(qd.reshape(batch_dim + (-1,)))
    nparray.append(x_pos.reshape(batch_dim + (-1,)))
    nparray.append(x_rot.reshape(batch_dim + (-1,)))
    nparray.append(xd_vel.reshape(batch_dim + (-1,)))
    nparray.append(xd_ang.reshape(batch_dim + (-1,)))
    nparray.append(xi_pos.reshape(batch_dim + (-1,)))
    nparray.append(xi_rot.reshape(batch_dim + (-1,)))
    nparray.append(xdi_vel.reshape(batch_dim + (-1,)))
    nparray.append(xdi_ang.reshape(batch_dim + (-1,)))
    nparray.append(j_pos.reshape(batch_dim + (-1,)))
    nparray.append(j_rot.reshape(batch_dim + (-1,)))
    nparray.append(jd_vel.reshape(batch_dim + (-1,)))
    nparray.append(jd_ang.reshape(batch_dim + (-1,)))
    nparray.append(ap_pos.reshape(batch_dim + (-1,)))
    nparray.append(ap_rot.reshape(batch_dim + (-1,)))
    nparray.append(ac_pos.reshape(batch_dim + (-1,)))
    nparray.append(ac_rot.reshape(batch_dim + (-1,)))
    nparray.append(mass.reshape(batch_dim + (-1,)))
    return jnp.concatenate(nparray, axis=-1)
