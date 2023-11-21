import numpy as np 
import sys

sys.path.append('C://Users//cmack//OneDrive - University of Cincinnati//Desktop//Flight Mechanics//Python_Files//Assignment3')# one directory up
import parameters.simulation_parameters as P
from tools.transformations import FrameTrans as Tran
from control.matlab import *

# import sys
# sys.path.append('C:/Users/cmack/OneDrive - University of Cincinnati/Desktop/Flight Mechanics/python-control-main/control')
# import matlab.timeresp 

class MisDynamics:
    def __init__(self, states0):
        # Initial state conditions
        self.state = np.array([
            [states0[0][0]],  # initial pn
            [states0[1][0]],  # initial pe
            [states0[2][0]],  # initial pd
            [states0[3][0]],  # initial u
            [states0[4][0]],  # initial v
            [states0[5][0]],  # initial w
            [states0[6][0]],  # initial phi
            [states0[7][0]],  # initial theta
            [states0[8][0]],  # initial psi
            [states0[9][0]],  # initial p
            [states0[10][0]], # initial q
            [states0[11][0]]])  # initial r                                    

        self.Ts = P.ts_simulation # Sim time step
        self.Jx = P.mi_jx            # MoI about the x-axis 
        self.Jy = P.mi_jy
        self.Jz = P.mi_jz
        self.Jxz = P.mi_jxz
        self.g = P.gravity        # Graviational constant
        self.m = P.mi_mass           # Mass of UAV

        self.G = P.mi_G
        self.G1 = P.mi_G1
        self.G2 = P.mi_G2
        self.G3 = P.mi_G3
        self.G4 = P.mi_G4
        self.G5 = P.mi_G5
        self.G6 = P.mi_G6
        self.G7 = P.mi_G7
        self.G8 = P.mi_G8

        # self.force_limit = P.F_max

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        # u = saturate(u, self.force_limit)
        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state, U):
        # Return xdot = f(x,u)
        pn = state[0][0]
        pe = state[1][0]
        pd = state[2][0]
        u = state[3][0]
        v = state[4][0]
        w = state[5][0]
        phi = state[6][0]
        theta = state[7][0]
        psi = state[8][0]
        p = state[9][0]
        q = state[10][0]
        r = state[11][0]

        Fx = U[0][0]
        Fy = U[1][0]
        Fz = U[2][0]
        L = U[3][0]
        M = U[4][0]
        N = U[5][0]

        # The equations of motion.

        statedot = np.array([[u*np.cos(theta)*np.cos(psi) + v*(np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi)) + w*(np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))],
                      [u*np.cos(theta)*np.sin(psi) + v*(np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi)) + w*(np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))],
                      [-u*np.sin(theta) + v*np.sin(phi)*np.cos(theta) + w*np.cos(phi)*np.cos(theta)],
                      [r*v - q*w + Fx/self.m],
                      [p*w - r*u + Fy/self.m],
                      [q*u - p*v + Fz/self.m],
                      [p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],
                      [q*np.cos(phi) - r*np.sin(phi)],
                      [q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)],
                      [self.G1*p*q - self.G2*q*r + self.G3*L + self.G4*N],
                      [self.G5*p*r - self.G6*(p**2 - r**2) + M/self.Jy],
                      [self.G7*p*q - self.G1*q*r + self.G4*L + self.G8*N]])

        pndot = statedot[0][0]
        pedot = statedot[1][0]
        pddot = statedot[2][0]
        udot = statedot[3][0]
        vdot = statedot[4][0]
        wdot = statedot[5][0]
        phidot = statedot[6][0]
        thetadot = statedot[7][0]
        psidot = statedot[8][0]
        pdot = statedot[9][0]
        qdot = statedot[10][0]
        rdot = statedot[11][0]

        # print("pddot: ", pddot)
        # print("psidot: ", psidot)

        # build xdot and return
        xdot = np.array([[pndot], [pedot], [pddot], [udot], [vdot], [wdot], [phidot], [thetadot], [psidot], [pdot], [qdot], [rdot]])
        return xdot

    def h(self):
        # return y = h(x)
        pn = self.state[0][0]
        pe = self.state[1][0]
        pd = self.state[2][0]
        u = self.state[3][0]
        v = self.state[4][0]
        w = self.state[5][0]
        phi = self.state[6][0]
        theta = self.state[7][0]
        psi = self.state[8][0]
        p = self.state[9][0]
        q = self.state[10][0]
        r = self.state[11][0]

        y = np.array([[pn], [pe], [pd], [u], [v], [w], [phi], [theta], [psi], [p], [q], [r]])
        return y

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts/2.*F1, u)
        F3 = self.f(self.state + self.Ts/2.*F2, u)
        F4 = self.f(self.state + self.Ts*F3, u)
        self.state += self.Ts/6. * (F1 + 2.*F2 + 2.*F3 + F4)

    def calcFPA(self, states, u):
        Vg = np.array([[states[3][0]], [states[4][0]], [states[5][0]]])
        VgMag = np.linalg.norm(Vg)

        hdot = -self.f(states, u)[2][0]
        FPA = np.arcsin(hdot/VgMag)

        return FPA
        
def saturate(u, limit):
    for i in range(0,len(u)):
        for j in range(0,len(u[0])):
            if abs(u[i][j]) > limit:
                u[i][j] = limit*np.sign(u[i][j])
    return u

class forces_moments:

    def aspeed(state, Vwss, Vwg):
        '''Take the current state variables, array of steady-state 
        wind components in vehicle frame, and array of gust wind
        components in body frame and calculate airspeed in body frame'''

        u = state[3][0]; v = state[4][0]; w = state[5][0]
        phi = state[6][0]; theta = state[7][0]; psi = state[8][0]

        # Find Windspeed and Angles for Lift, Drag, and Sideslip
        Vg = np.array([[u], [v], [w]])              #Ground speed in Body Frame
        Vw = Tran.v2b(Vwss, phi, theta, psi) + Vwg  #Wind velocity in Body Frame
        Va = Vg - Vw                                #Airspeed in body frame

        return Va

    def forces(state, d):
        '''Take the current state, Array of control surface deflections 
            [aileron, elevator, rudder, prop], and Airspeed in body frame
            and compute net x, y, and z forces on the aircraft in body frame'''
        
        # States
        u = state[3][0];    v = state[4][0];        w = state[5][0]
        phi = state[6][0];  theta = state[7][0];    psi = state[8][0]
        p = state[9][0];    q = state[10][0];       r = state[11][0]

        # Inputs
        d_ail = d[0][0]
        d_ele = d[1][0]
        d_rud = d[2][0]
        d_t = d[3][0]

        # Parameters
        m = P.mass;     g = P.gravity;      rho = P.rho
        S = P.S_wing;   c = P.c;            AR = P.AR;  b = P.b


        # Find the force of Gravity in body frame
        Fg = Tran.v2b([[0.],[0.],[m*g]], phi, theta, psi)

        # Find the magnitude of airspeed and Wind Frame Angles
        Va = np.array([[u], [v], [w]])              # Airspeed in body frame WHEN NO WIND -- CHANGE AFTER HW5
        VaMag = np.linalg.norm(Va)                  #Magnitude of airspeed
        alpha = np.arctan2(Va[2][0],Va[0][0])           #Angle of Attack
        beta = np.arcsin(Va[1][0]/VaMag)                #Sideslip Angle


        # Find the force of Lift
        CL0 = P.C_L_0;  CLa = P.C_L_alpha;  CLq = P.C_L_q;  CLde = P.C_L_delta_e
        M = P.M;        aL0 = P.alpha0

        sig = (1 + np.exp(-M*(alpha-aL0)) + np.exp(M*(alpha+aL0))) / ((1 + np.exp(-M*(alpha-aL0)))*(1 + np.exp(M*(alpha+aL0))))
        FL = 0.5*rho*S*VaMag**2 * (CLq*c*q/(2*VaMag) + CLde*d_ele + (1-sig)*(CL0+CLa*alpha) + sig*(2*np.sign(alpha)*np.cos(alpha)*np.sin(alpha)**2))  #Magnitude of Lift
        FL = np.array([[0.], [0.], [-FL]])   #Force of Lift in Stability Frame (since kw = ks)
        FL = Tran.s2b(FL, alpha)            #Force of Lift in Body Frame


        # Find the force of Drag
        CDp = P.C_D_p;  e = P.e

        FD = 0.5*rho*S*VaMag**2 * (CDp + (CL0 + CLa*alpha)**2 /(np.pi*e*AR))  #Magnitude of Drag Force
        FD = np.array([[-FD], [0.], [0.]])  #Force of Drag in Stability Frame (Fs deals with sideslip force)
        FD = Tran.s2b(FD, alpha)            #Force of Drag in Body Frame

        # Find the Sideslip Force
        CY0 = P.C_Y_0;  CYB = P.C_Y_beta;   CYp = P.C_Y_p;  CYr = P.C_Y_r
        CYda = P.C_Y_delta_a;               CYdr = P.C_Y_delta_r

        Fs = 0.5*rho*S*VaMag**2 * (CY0 + CYB*beta + CYp*b*p/(2*VaMag) + CYr*b*r/(2*VaMag) + CYda*d_ail + CYdr*d_rud)
        Fs = np.array([[0.], [Fs], [0.]])   #Sideslip Force in Wind Frame
        Fs = Tran.w2s(Fs, beta)      #Sideslip Force in Body Frame (since js = jb)


        # Find the Force of Thrust
        Sprop = P.S_prop; Cprop = P.C_prop; km = P.k_motor

        Fprop = 0.5*rho*Sprop*Cprop*((km*d_t)**2 - VaMag**2)    #Magnitude of propellor force
        Fprop = np.array([[Fprop], [0], [0]])                   #Propellor Force in Body Frame


        # Find the Total Force
        Ftot = FL + FD + Fprop + Fs + Fg
        fx = Ftot[0][0]
        fy = Ftot[1][0]
        fz = Ftot[2][0]

        return fx, fy, fz
        
    def moments(state, d):
        '''Take the current state, the Steady State Wind Component,
            The Wind Gust Component, and the control surface deflection
            [aileron, elevator, rudder, prop]'''

        # States
        u = state[3][0];    v = state[4][0];        w = state[5][0]
        phi = state[6][0];  theta = state[7][0];    psi = state[8][0]
        p = state[9][0];    q = state[10][0];       r = state[11][0]

        # Inputs
        d_ail = d[0][0]
        d_ele = d[1][0]
        d_rud = d[2][0]
        d_t = d[3][0]

        # Parameters
        rho = P.rho
        S = P.S_wing;   c = P.c;    b = P.b


        # Find Airspeed, Angle of Attack, and Sideslip Angle
        Va = np.array([[u], [v], [w]])                  # Airspeed when no wind -- DELETE AFTER HW5
        VaMag = np.linalg.norm(Va)                      #Magnitude of airspeed
        alpha = np.arctan2(Va[2][0],Va[0][0])           #Angle of Attack
        beta = np.arcsin(Va[1][0]/VaMag)                #Sideslip Angle


        # Find Pitching Moment
        Cm0 = P.C_m_0;  Cma = P.C_m_alpha;  Cmq = P.C_m_q;  Cmde = P.C_m_delta_e

        mp = 0.5*rho*S*c*VaMag**2 * (Cm0 + Cma*alpha + Cmq*c*q/(2*VaMag) + Cmde*d_ele)  #Magnitude of Pitching Moment
        mp = np.array([[0.], [mp], [0.]])   # Pitching Moment in Body Frame


        # Find Rolling Moment
        Cl0 = P.C_ell_0;  ClB = P.C_ell_beta;   Clp = P.C_ell_p;    Clr = P.C_ell_r;   
        Clda = P.C_ell_delta_a;                 Cldr = P.C_ell_delta_r

        ml = 0.5*rho*S*b*VaMag**2 * (Cl0 + ClB*beta + Clp*b*p/(2*VaMag) + Clr*b*r/(2*VaMag) + Clda*d_ail + Cldr*d_rud)  #Magnitude of Rolling Moment
        ml = np.array([[ml], [0.], [0.]])   # Rolling Moment in Body Frame (I think it's supposed to be in Body?)


        # Find Yawing Moment
        Cn0 = P.C_n_0;  CnB = P.C_n_beta;   Cnp = P.C_n_p;  Cnr = P.C_n_r;
        Cnda = P.C_n_delta_a;               Cndr = P.C_n_delta_r

        mn = 0.5*rho*S*b*VaMag**2 * (Cn0 + CnB*beta + Cnp*b*p/(2*VaMag) + Cnr*b*r/(2*VaMag) + Cnda*d_ail + Cndr*d_rud)  #Magnitude of Yawing Moment
        mn = np.array([[0.], [0.], [mn]])   # Yawing Moment in Body Frame (I think it's body frame?)


        # Find Propellor Moment
        kTp = P.k_Tp;   kO = P.k_omega

        mprop = -kTp*(kO*d_t)**2    #Magnitude of Propellor Reaction Moment
        mprop = np.array([[mprop], [0.], [0.]])


        # Find Total Moments
        Moms = mp + ml + mn + mprop
        L = Moms[0][0]
        M = Moms[1][0]
        N = Moms[2][0]

        return L, M, N
    
class Gusts:
    def Dryden(state, Lu, Lv, Lw, Su, Sv, Sw, Va):
        '''Compute the components of the gust disturbance'''
        dt = P.ts_simulation
        phi = state[6][0]; theta = state[7][0]; psi = state[8][0]

        Hunum = [0.,Su*np.sqrt(2.*Va/Lu)]
        Huden = [1., Va/Lu]
        Hu = tf(Hunum,Huden)

        Hvnum = np.multiply(Sv*np.sqrt(3.*Va/Lv),[0., 1., Va/(np.sqrt(3.)*Lv)])
        Hvden = [1., 2.*Va/Lv, (Va/Lv)**2]
        Hv = tf(Hvnum, Hvden)

        Hwnum = np.multiply(Sw*np.sqrt(3*Va/Lw),[0., 1., Va/(np.sqrt(3.)*Lw)])
        Hwden = [1., 2.*Va/Lw, (Va/Lw)**2]
        Hw = tf(Hwnum, Hwden)

        WN_u = np.random.normal(0,1,1)
        WN_v = np.random.normal(0,1,1)
        WN_w = np.random.normal(0,1,1)

        y_u, Tu, x_u = lsim(Hu, U=WN_u[0], T=[0,dt])
        y_v, Tv, x_v = lsim(Hv, U=WN_v[0], T=[0,dt])
        y_w, Tw, x_v = lsim(Hw, U=WN_w[0], T=[0,dt])

        Vwgx = y_u[1]
        Vwgy = y_v[1]
        Vwgz = y_w[1]

        Vwg = np.array([[Vwgx], [Vwgy], [Vwgz]])  #Gust Vector in Body Frame
        Vwg = Tran.b2v(Vwg, phi, theta, psi)      #Gust Vector in Vehicle Frame
        return Vwg
