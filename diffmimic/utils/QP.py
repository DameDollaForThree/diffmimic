class QP():
    def __init__(self, q, qd, x_pos, x_rot, xd_vel, xd_ang,
                 xi_pos, xi_rot, xdi_vel, xdi_ang,
                 j_pos, j_rot, jd_vel, jd_ang,
                 ap_pos, ap_rot, ac_pos, ac_rot, mass) -> None:
        self.q = q
        self.qd = qd
        self.x_pos = x_pos
        self.x_rot = x_rot
        self.xd_vel = xd_vel
        self.xd_ang = xd_ang
        self.xi_pos = xi_pos
        self.xi_rot = xi_rot
        self.xdi_vel = xdi_vel
        self.xdi_ang = xdi_ang
        self.j_pos = j_pos
        self.j_rot = j_rot
        self.jd_vel = jd_vel
        self.jd_ang = jd_ang
        self.ap_pos = ap_pos
        self.ap_rot = ap_rot
        self.ac_pos = ac_pos
        self.ac_rot = ac_rot
        self.mass = mass