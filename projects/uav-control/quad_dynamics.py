#!/bin/usr/env python3 

import numpy as np 

class Quad():
    def __init__(self, mass):
        """
        pos, vel, acc, jerk, x_w_B, y_w_B, z_w_B are all numpy arrays of shape (3,)
        mass, phi, phidot, and u are floats
        """

        # Dynamics
        self.mass_ = mass
        self.g_ = 9.81

        # State and input thrust
        self.pos_ = None
        self.vel_ = None
        self.acc_ = None
        self.jerk_ = None
        self.phi_ = None
        self.phidot_ = None
        self.u_ = None

        # Body coordinate frame expressed in world frame 
        self.x_w_B = None
        self.y_w_B = None
        self.z_w_B = None
        self.R_w_B = None

    def get_state(self, pos, vel, acc, jerk, phi, phidot):
        self.pos_ = pos
        self.vel_ = vel
        self.acc_ = acc
        self.phi_ = phi
        self.phidot_ = phidot
        self.u_ = u

    def get_body_in_world(self):
        """
        Computes coordinates of body frame in world frame. 
        """

        # Compute z 
        v = self.acc_ + np.array([0, 0, self.g_])
        self.z_w_B = v / np.linalg.norm(v) 

        # Compute y
        x_aux = np.array([np.cos(self.phi_), np.sin(self.phi_), 0])
        cross_prod = np.cross(self.z_w_B, x_aux)
        self.y_w_B = cross_prod / np.linalg.norm(cross_prod)

        # Compute x 
        self.x_w_B = np.cross(self.y_w_B, self.z_w_B)

        # Compute rotation matrix from body to world 
        self.R_w_B = np.hstack((self.x_w_B, self.y_w_B, self.z_w_B))

    
    def compute_h_omega(self):
        if self.pos_ and self.jerk_ and self.u_:
            return (self.mass/self.u_) * (self.jerk_ - (self.z_w_B @ self.jerk_) * self.z_w_B)
        else:
            raise Exception("Input, Positoin, and Jerk fields shouldn't be empty!")

    def compute_omega_world(self):
        # get h_omega
        h_omega = self.compute_h_omega()

        # compute p, q, and r
        p = -h_omega @ self.y_w_B
        q = h_omega @ self.x_w_B
        r = self.phidot_ * self.z_w_B[2]

        return p * self.x_w_B + q * self.y_w_B + r * self.z_w_B 

    def compute_omega_body(self)
        omega_world = self.compute_omega_world()
        try: 
            R_B_w = np.linalg.inv(self.R_w_B)
            return R_B_w @ omega_world 
        except np.linalg.LinAlgError as e:
            print("Matrix is not invertible:", e)
