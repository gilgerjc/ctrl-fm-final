import numpy as np

class FrameTrans:
    def v2b(I, phi, theta, psi):
        '''Converts a 3xn matrix from vehicle frame to body frame'''

        R = np.array([[np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
                    [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
                    [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]])

        B = np.matmul(R,I)

        return B

    def b2v(B, phi, theta, psi):
        '''Converts a 3xn matrix from body frame to vehicle/inertial frame'''

        R = np.array([[np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
                    [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
                    [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]])

        R = np.transpose(R)

        I = np.matmul(R,B)

        return I

    def b2s(B, alp):
        '''Converts a 3xn matrix from body frame to stability frame'''

        R = np.array([[np.cos(alp), 0., np.sin(alp)],
                    [0., 1., 0.],
                    [-np.sin(alp), 0., np.cos(alp)]])

        S = np.matmul(R,B)

        return S

    def s2b(S, alp):
        '''Converts a 3xn matrix from stability frame to body frame'''

        R = np.array([[np.cos(alp), 0., np.sin(alp)],
                    [0., 1., 0.],
                    [-np.sin(alp), 0., np.cos(alp)]])

        R = np.transpose(R)

        B = np.matmul(R,S)

        return B

    def s2w(S,beta):
        '''Converts a 3xn matrix from stability frame to wind frame'''

        R = np.array([[np.cos(beta), np.sin(beta), 0.],
                    [-np.sin(beta), np.cos(beta), 0.],
                    [0., 0., 1.]])

        W = np.matmul(R,S)

        return W

    def w2s(W,beta):
        '''Converts a 3xn matrix from stability frame to wind frame'''

        R = np.array([[np.cos(beta), np.sin(beta), 0.],
                    [-np.sin(beta), np.cos(beta), 0.],
                    [0., 0., 1.]])

        R = np.transpose(R)

        S = np.matmul(R,W)

        return S

    def b2w(B, alp, beta):
        '''Converts a 3xn matrix from body frame directly to wind frame'''

        R = np.array([[np.cos(beta)*np.cos(alp), np.sin(beta), np.cos(beta)*np.sin(alp)],
                    [-np.sin(beta)*np.cos(alp), np.cos(beta), -np.sin(beta)*np.sin(alp)],
                    [-np.sin(alp), 0., np.cos(alp)]])

        W = np.matmul(R,B)

        return W

    def w2b(W, alp, beta):
        '''Converts a 3xn matrix from wind frame directly to body frame'''

        R = np.array([[np.cos(beta)*np.cos(alp), np.sin(beta), np.cos(beta)*np.sin(alp)],
                    [-np.sin(beta)*np.cos(alp), np.cos(beta), -np.sin(beta)*np.sin(alp)],
                    [-np.sin(alp), 0., np.cos(alp)]])

        R = np.transpose(R)

        B = np.matmul(R,W)

        return B