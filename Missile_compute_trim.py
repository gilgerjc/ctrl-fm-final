import sys
sys.path.append('C://Users//cmack//OneDrive - University of Cincinnati//Desktop//Flight Mechanics//Python_Files//Assignment4')
import numpy as np

# load message types
#from message_types.msg_state import MsgState
from tools.rotations import Quaternion2Euler, Quaternion2Rotation, Euler2Rotation
from math import cos, sin, tan
# import control
from control.matlab import *
import parameters.simulation_parameters as P
from dynamics.Missile_Dynamics import MisDynamics
from dynamics.Missile_Dynamics import forces_moments as FM
from scipy.optimize import minimize


class ComputeTrim:

    def __init__(self):
        # self.P=P
        self.Ts=P.ts_simulation
        self.mav = MisDynamics(P.mis_states0)
        
    def compute_trim(self, Va, Y, R):
        '''Take current state, desired Va magnitude, target Flight Path Angle,
        and target Turn Radius, and compute the trim states and trim inputs
        Outputs in the form x = [pn,pe,pd,u,v,w,phi,theta,psi,p,q,r] and
        u = [d_ail, d_ele, d_rud, d_t]'''

        x0 = np.array([0,0,0])
        res = minimize(lambda x: self.compute_trim_cost(x,Va,Y,R), x0, method='nelder-mead',options={'xatol': 1e-9, 'disp': True})
        x_trim, u_trim = self.compute_trim_states_input(res.x,Va,Y,R)   # Find trim states and trim deflections
        return (x_trim, u_trim)         # Return Trim States and Trim Inputs

    def compute_trim_states_input(self, x, Va, Y, R):
        '''Take wing angles (alpha, beta, phi), desired Aispeed magnitude,
        target Flight Path Angle, and target Turn Radius'''

        # Inertial parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        jx = P.mi_jx;  jy = P.mi_jy;  jz = P.mi_jz;  jxz = P.mi_jxz
        G = P.mi_G;    G1 = P.mi_G1;  G2 = P.mi_G2;  G3 = P.mi_G3
        G4 = P.mi_G4;  G5 = P.mi_G5;  G6 = P.mi_G6;  G7 = P.mi_G7;  G8 = P.mi_G8

        g = P.gravity
        m = P.mi_mass

        ## aerodynamic parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        S_wing        = P.mi_S;   b = P.mi_b; c = P.mi_c
        
        rho           = P.rho
        e             = P.e
        AR            = P.pl_AR
        M             = P.M 
        alpha0        = P.alpha0
        epsilon       = P.epsilon

        C_L_0         = P.mi_C_L_0;        C_D_0 = P.mi_C_D_0;                C_m_0 = P.mi_C_m_0
        C_L_alpha     = P.mi_C_L_alpha;    C_D_alpha = P.mi_C_D_alpha;        C_m_alpha = P.mi_C_m_alpha
        C_L_q         = P.mi_C_L_q;        C_D_q = P.mi_C_D_q;                C_m_q = P.mi_C_m_q
        C_L_delta_e   = P.mi_C_L_delta_e;  C_D_delta_e = P.mi_C_D_delta_e;    C_m_delta_e = P.mi_C_m_delta_e
    
        C_D_p         = P.mi_C_D_p
        C_Y_0         = P.mi_C_Y_0;        C_ell_0 = P.mi_C_ell_0;            C_n_0 = P.mi_C_n_0
        C_Y_beta      = P.mi_C_Y_beta;     C_ell_beta = P.mi_C_ell_beta;      C_n_beta = P.mi_C_n_beta 
        
        C_Y_p         = P.mi_C_Y_p;        C_ell_p = P.mi_C_ell_p;            C_n_p = P.mi_C_n_p
        C_Y_r         = P.mi_C_Y_r;        C_ell_r = P.mi_C_ell_r;            C_n_r = P.mi_C_n_r
        
        C_Y_delta_a   = P.mi_C_Y_delta_a;  C_ell_delta_a = P.mi_C_ell_delta_a; C_n_delta_a = P.mi_C_n_delta_a
        C_Y_delta_r   = P.mi_C_Y_delta_r;  C_ell_delta_r = P.mi_C_ell_delta_r; C_n_delta_r = P.mi_C_n_delta_r
        
        k_th       = P.pl_k_th
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Wind Angles
        alpha=x[0]
        beta=x[1]
        phi=x[2]

        # Compute necessary u, v, w, theta to maintain trim condition
        u = Va*np.cos(alpha)*np.cos(beta)
        v = Va*np.sin(beta)
        w = Va*np.sin(alpha)*np.cos(beta)
        theta = alpha + Y
        
        # Compute necessary p, q, r to maintain trim
        p = (-Va/R)*np.sin(theta)
        q = (Va/R)*np.sin(phi)*np.cos(theta)
        r = (Va/R)*np.cos(phi)*np.cos(theta)
        
        # Assemble calculated trim states into x vector
        x_trim=np.array([[0],[0],[0],[u],[v],[w],[phi],[theta],[0],[p],[q],[r]])

        # Find lift and drag coefficients with alpha incorporated
        C_L = C_L_0 + C_L_alpha*alpha
        C_D = C_D_0 + C_D_alpha*alpha
        
        # Find Cx, Cxq, CxDELe with alpha incorporated
        C_X = -C_D*np.cos(alpha) + C_L*np.sin(alpha)
        C_X_q = -C_D_q*np.cos(alpha) + C_L_q*np.sin(alpha)
        C_X_delta_e = -C_D_delta_e*np.cos(alpha) + C_L_delta_e*np.sin(alpha)
        
        # Find Cz, Czq, CzDELe with alpha incorporated
        C_Z = -C_D*np.sin(alpha) - C_L*np.cos(alpha)
        C_Z_q = -C_D_q*np.sin(alpha) - C_L_q*np.cos(alpha)
        C_Z_delta_e = -C_D_delta_e*np.sin(alpha) - C_L_delta_e*np.cos(alpha)

        # Find necessary elevator deflection to maintain trim
        d_e = (((jxz*(p**2-r**2)+(jx-jz)*p*r)/(0.5*rho*(Va**2)*c*S_wing))-C_m_0-C_m_alpha*alpha-C_m_q*((c*q)/(2.*Va)))/C_m_delta_e

        # Find necessary d_t to maintain trim
        d_t = (m*(-r*v+q*w+g*np.sin(theta))-rho*(Va**2)*S_wing*(C_X+C_X_q*((c*q)/(2*Va))+C_X_delta_e*d_e))/k_th

        # Find necessary rudder and aileron deflection to maintain trim
        temp_1=np.linalg.inv(np.array([[C_ell_delta_a, C_ell_delta_r],
                        [C_n_delta_a, C_n_delta_r]]))
        temp_2=np.array([[((-G1*p*q+G2*q*r)/(0.5*rho*(Va**2)*S_wing*b))-C_ell_0-C_ell_beta*beta-C_ell_p*((b*p)/(2*Va))-C_ell_r*((b*r)/(2*Va))],
                        [((-G7*p*q+G1*q*r)/(0.5*rho*(Va**2)*S_wing*b))-C_n_0-C_n_beta*beta-C_n_p*((b*p)/(2*Va))-C_n_r*((b*r)/(2*Va))]])
        temp_3=np.matmul(temp_1,temp_2)
         
        d_a = np.asscalar(temp_3[0])
        d_r = np.asscalar(temp_3[1])
        
        u_trim=np.array([[d_a],[d_e],[d_r],[d_t]])
        return (x_trim, u_trim)
    
    def compute_trim_cost(self, x, Va, Y, R):
        '''Takes Wind Angles, desired Airpseed magnitude,
        target Flight Path Angle, and target Turn Radius and calculates 
        how much change is necessary for Trim'''
        
        # Calc magnitude of airspeed for later calcs
        # Va = np.linalg(VaVec)

        # Wind Angles
        phi=x[0]
        alpha=x[1]
        beta=x[2]

        #Va=35
        #R=99999999999
        #Y=0

        # Compute trim states and inputs that create trim conditions
        x_trim, d_trim = self.compute_trim_states_input(x,Va,Y,R)
        
        #f_x, f_y, f_z, tau_phi, tau_theta, tau_psi=forces_moments(x_trim, d_e, d_a, d_r, d_t)
        fx, fy, fz = FM.forces(x_trim, d_trim)
        L, M, N = FM.moments(x_trim, d_trim)
        ForMom = np.array([[fx],[fy],[fz],[L],[M],[N]])

        #print('fx=',f_x,'fy=', f_y, 'fz=', f_z, 'l=',tau_phi, 'm=',tau_theta, 'n=',tau_psi)
        # U=f_m
        #U=np.array([f_x,f_y,f_z,tau_phi,tau_theta,tau_psi])

        states_dot= self.mav.f(x_trim, ForMom) # 

        # Compute X_dot_star (rate of change of trim states)
        x_dot=np.array([[0],
                        [0],
                        [-Va*sin(Y)], # I am using Pd_dot not hdot..that is why there is a sign change
                        [0.],
                        [0.],
                        [0.],
                        [0.],
                        [0.],
                        [Va*np.cos(Y)/R],
                        [0.],
                        [0.],
                        [0.]])
        
        #trimmed_inputs=np.array([d_e,d_t,d_a,d_r])

        J=np.linalg.norm(x_dot-states_dot)**2
        return J
    
class compute_gains:
    def compute_tf_models(x_trim, u_trim):
    
        # Inertial parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        jx = P.mi_jx;  jy = P.mi_jy;  jz = P.mi_jz;  jxz = P.mi_jxz
        G = P.mi_G;    G1 = P.mi_G1;  G2 = P.mi_G2;  G3 = P.mi_G3
        G4 = P.mi_G4;  G5 = P.mi_G5;  G6 = P.mi_G6;  G7 = P.mi_G7;  G8 = P.mi_G8

        g = P.gravity
        m = P.mi_mass

        ## aerodynamic parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        S_wing        = P.mi_S;   b = P.mi_b; c = P.mi_c
        
        rho           = P.rho
        e             = P.e
        AR            = P.pl_AR
        M             = P.M 
        alpha0        = P.alpha0
        epsilon       = P.epsilon

        C_L_0         = P.mi_C_L_0;        C_D_0 = P.mi_C_D_0;                C_m_0 = P.mi_C_m_0
        C_L_alpha     = P.mi_C_L_alpha;    C_D_alpha = P.mi_C_D_alpha;        C_m_alpha = P.mi_C_m_alpha
        C_L_q         = P.mi_C_L_q;        C_D_q = P.mi_C_D_q;                C_m_q = P.mi_C_m_q
        C_L_delta_e   = P.mi_C_L_delta_e;  C_D_delta_e = P.mi_C_D_delta_e;    C_m_delta_e = P.mi_C_m_delta_e
    
        C_D_p         = P.mi_C_D_p
        C_Y_0         = P.mi_C_Y_0;        C_ell_0 = P.mi_C_ell_0;            C_n_0 = P.mi_C_n_0
        C_Y_beta      = P.mi_C_Y_beta;     C_ell_beta = P.mi_C_ell_beta;      C_n_beta = P.mi_C_n_beta 
        
        C_Y_p         = P.mi_C_Y_p;        C_ell_p = P.mi_C_ell_p;            C_n_p = P.mi_C_n_p
        C_Y_r         = P.mi_C_Y_r;        C_ell_r = P.mi_C_ell_r;            C_n_r = P.mi_C_n_r
        
        C_Y_delta_a   = P.mi_C_Y_delta_a;  C_ell_delta_a = P.mi_C_ell_delta_a; C_n_delta_a = P.mi_C_n_delta_a
        C_Y_delta_r   = P.mi_C_Y_delta_r;  C_ell_delta_r = P.mi_C_ell_delta_r; C_n_delta_r = P.mi_C_n_delta_r
        
        k_th       = P.pl_k_th
        # End Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Va_trim = np.sqrt(x_trim[3][0]**2 +x_trim[4][0]**2 +x_trim[5][0]**2)
        alpha_trim = np.arctan2(x_trim[5][0],x_trim[3][0])
        beta_trim = np.arctan2(x_trim[4][0],np.sqrt(x_trim[3][0]**2 + x_trim[5][0]**2))
        theta_trim = x_trim[7][0]

        #$ define transfer function constants
        a_phi1   = -0.25*rho*Va_trim*S_wing*b**2*C_ell_p
        a_phi2   = 0.5*rho*Va_trim**2*S_wing*b*C_ell_delta_a
        
        # a_beta1  = -rho*Va_trim*S_wing*C_Y_beta/(2*m*np.cos(beta_trim))
        # a_beta2  = rho*Va_trim*S_wing*C_Y_delta_r/(2*m*np.cos(beta_trim))
        
        a_theta1 = -rho*Va_trim*c**2*S_wing*C_m_q/(4*P.mi_jy)
        a_theta2 = -rho*Va_trim**2*c*S_wing*C_m_alpha/(2*P.mi_jy)
        a_theta3 = rho*Va_trim**2*c*S_wing*C_m_delta_e/(2*P.mi_jy)

        # a_V1     = rho*Va_trim*S_wing*(C_D_0 + C_D_alpha*alpha_trim + C_D_delta_e*u_trim[1][0])/P.mass + rho*S_prop*P.C_prop*Va_trim/P.mass
        # a_V2     = rho*S_prop*P.C_prop*P.k_motor**2*u_trim[3][0]/P.mass
        # a_V3     = g
            

        # Gains from Transfer Functions
        kp_ph = P.da_max/P.phie_max
        kd_ph = (2*P.zph*P.wn_ph-a_phi1)/a_phi2

        kp_ch = 2*P.wn_ch*P.zch*Va_trim/P.gravity
        ki_ch = P.wn_ch**2*Va_trim/P.gravity

        # kp_be = P.dr_max/P.bete_max
        # ki_be = ((a_beta1 + a_beta2*kp_be)/(2*P.zbe))**2/a_beta2

        kp_th = P.de_max/P.thee_max*np.sign(a_theta3)
        wn_th = np.sqrt(a_theta2+kp_th*a_theta3)
        kd_th = (2*P.zth*wn_th - a_theta1)/a_theta3
        kDC_th = (kp_th*a_theta3)/(a_theta2 + kp_th*a_theta3)

        # wn_h = wn_th/P.Wh
        # kp_h = (2*P.zh*wn_h)/(kDC_th*Va_trim)
        # ki_h = wn_h**2/(kDC_th*Va_trim)

        # wn_v2 = wn_th/P.Wv2
        # kp_v2 = (a_V1 - 2*P.zv2*wn_v2)/(kDC_th*P.gravity)
        # ki_v2 = wn_v2**2/(kDC_th*P.gravity)

        # kp_V = (2*P.zV*P.wn_V - a_V1)/a_V2
        # ki_V = P.wn_V**2/a_V2

        ph_gs = np.array([kp_ph, kd_ph, 0.])
        ch_gs = np.array([kp_ch, 0., ki_ch])
        # be_gs = np.array([kp_be, 0., ki_be])
        th_gs = np.array([kp_th, kd_th, 0.])
        # h_gs = np.array([kp_h, 0., ki_h])
        # V2_gs = np.array([kp_v2, 0., ki_v2])
        # V_gs = np.array([kp_V, 0., ki_V])

        # Return a 7x3 array of gains. Indeces:
        # 1: phi hold   2: chi hold   3: beta hold
        # 4: theta hold  5: h hold  6: Aspeed hold via pitch  7: Aspeed hold via throttle
        return (ph_gs, ch_gs, th_gs)
    
class StateSpace:
    def Lateral(x_trim, u_trim):
        Va_trim = np.sqrt(x_trim[3][0]**2 +x_trim[4][0]**2 +x_trim[5][0]**2)
        beta_trim = np.arctan2(x_trim[4][0],np.sqrt(x_trim[3][0]**2 + x_trim[5][0]**2))
        
        d_a = u_trim[0][0]; d_r = u_trim[3][0]
        u_trim = x_trim[3][0]; v_trim = x_trim[4][0]; w_trim = x_trim[5][0]
        phi_trim = x_trim[6][0]; theta_trim = x_trim[7][0]; psi_trim = x_trim[8][0]
        p_trim = x_trim[9][0]; q_trim = x_trim[10][0]; r_trim = x_trim[11][0]

        Yv = P.rho*P.S_wing/P.mass*(P.b*v_trim*(P.C_Y_p*p_trim+P.C_Y_r*r_trim)/(4*Va_trim) + v_trim*(P.C_Y_0+P.C_Y_beta*beta_trim+P.C_Y_delta_a*d_a+P.C_Y_delta_r*d_r) + P.C_Y_beta*np.sqrt(u_trim**2+w_trim**2)/2.)
        Yp = w_trim + P.rho*Va_trim*P.S_wing*P.b*P.C_Y_p/(4.*P.mass)
        Yr = -u_trim + P.rho*Va_trim*P.S_wing*P.b*P.C_Y_r/(4.*P.mass)
        Yda = P.rho*Va_trim**2*P.S_wing*P.C_Y_delta_a/(2.*P.mass)
        Ydr = P.rho*Va_trim**2*P.S_wing*P.C_Y_delta_r/(2.*P.mass)

        Lv = P.rho*P.S_wing*P.b*(P.b*v_trim*(P.C_p_p*p_trim+P.C_p_r*r_trim)/(4.*Va_trim) + v_trim*(P.C_p_0+P.C_p_beta*beta_trim+P.C_p_da*d_a+P.C_p_dr*d_r) + P.C_p_beta/2.*np.sqrt(u_trim**2+w_trim**2))
        Lp = P.G1*q_trim + P.rho*Va_trim*P.S_wing*P.b**2*P.C_p_p/4.
        Lr = -P.G2*q_trim + P.rho*Va_trim*P.S_wing*P.b**2*P.C_p_r/4.
        Lda = P.rho*Va_trim**2*P.S_wing*P.b*P.C_p_da/2.
        Ldr = P.rho*Va_trim**2*P.S_wing*P.b*P.C_p_dr/2.

        Nv = P.rho*P.S_wing*P.b*(P.b*v_trim*(P.C_r_p*p_trim+P.C_r_r*r_trim)/(4.*Va_trim) + v_trim*(P.C_r_0+P.C_r_beta*beta_trim+P.C_r_da*d_a+P.C_r_dr*d_r) + P.C_r_beta/2.*np.sqrt(u_trim**2+w_trim**2))
        Np = P.G7*q_trim + P.rho*Va_trim*P.S_wing*P.b**2*P.C_r_p/4.
        Nr = -P.G1*q_trim + P.rho*Va_trim*P.S_wing*P.b**2*P.C_r_r/4.
        Nda = P.rho*Va_trim**2*P.S_wing*P.b*P.C_r_da/2.
        Ndr = P.rho*Va_trim**2*P.S_wing*P.b*P.C_r_dr/2.

        # Latxdot = [vdot, pdot, rdot, phidot, psidot]
        A = np.array([[Yv, Yp, Yr, P.gravity*np.cos(theta_trim)*np.cos(phi_trim), 0.],
                            [Lv, Lp, Lr, 0., 0.],
                            [Nv, Np, Nr, 0., 0.],
                            [0., 1., np.cos(phi_trim)*np.tan(theta_trim), q_trim*np.cos(phi_trim)*np.tan(theta_trim) - r_trim*np.sin(phi_trim)*np.tan(theta_trim), 0.],
                            [0., 0., np.cos(phi_trim)/np.cos(theta_trim), p_trim*np.cos(phi_trim)/np.cos(theta_trim) - r_trim*np.sin(phi_trim)/np.cos(theta_trim), 0.]])

        B = np.array([[Yda, Ydr],
                      [Lda, Ldr],
                      [Nda, Ndr],
                      [0., 0.],
                      [0., 0.]])
        
        x = np.array([[v_trim], [p_trim], [r_trim], [phi_trim], [psi_trim]])
        u = np.array([[d_a], [d_r]])
        Latxdot = np.matmul(A, x) + np.matmul(B, u)
        
        return Latxdot, A, B

    def Longitudinal(x_trim, u_trim):
        Va_trim = np.sqrt(x_trim[3][0]**2 +x_trim[4][0]**2 +x_trim[5][0]**2)
        alpha_trim = np.arctan2(x_trim[5][0],x_trim[3][0])
        
        d_e = u_trim[1][0]; d_t = u_trim[3][0]
        h_trim = -x_trim[2][0]
        u_trim = x_trim[3][0]; w_trim = x_trim[5][0]
        theta_trim = x_trim[7][0]
        q_trim = x_trim[10][0]
        
        C_X_0=-P.C_D_0*np.cos(alpha_trim)+P.C_L_0*np.sin(alpha_trim)
        C_X_alpha = -P.C_D_alpha*np.cos(alpha_trim)+P.C_L_alpha*np.sin(alpha_trim)
        C_X_q=-P.C_D_q*np.cos(alpha_trim)+P.C_L_q*np.sin(alpha_trim)
        C_X_de=-P.C_D_delta_e*np.cos(alpha_trim)+P.C_L_delta_e*np.sin(alpha_trim)

        C_Z_0=-P.C_D_0*np.sin(alpha_trim)-P.C_L_0*np.cos(alpha_trim)
        C_Z_alpha = -P.C_D_alpha*np.sin(alpha_trim)-P.C_L_alpha*np.cos(alpha_trim)
        C_Z_q=-P.C_D_q*np.sin(alpha_trim)-P.C_L_q*np.cos(alpha_trim)
        C_Z_de=-P.C_D_delta_e*np.sin(alpha_trim)-P.C_L_delta_e*np.cos(alpha_trim)

        Xu = P.rho*P.S_wing/P.mass*(u_trim*(C_X_0+C_X_alpha*alpha_trim+C_X_de*d_e) - w_trim*C_X_alpha/2. + P.c*C_X_q*u_trim*q_trim/(4.*Va_trim)) - P.rho*P.S_prop*P.C_prop*u_trim/P.mass
        Xw = -q_trim + P.rho*P.S_wing/P.mass*(w_trim*(C_X_0+C_X_alpha*alpha_trim+C_X_de*d_e) + P.c*C_X_q*w_trim*q_trim/(4.*Va_trim) + C_X_alpha*u_trim/2.) - P.rho*P.S_prop*P.C_prop*w_trim/P.mass
        Xq = -w_trim + P.rho*Va_trim*P.S_wing*C_X_q*P.c/(4.*P.mass)
        Xde = P.rho*Va_trim**2*P.S_wing*C_X_de/(2.*P.mass)
        Xdt = P.rho*P.S_prop*P.C_prop*P.k_motor**2*d_t/P.mass

        Zu = q_trim + P.rho*P.S_wing/P.mass*(u_trim*(C_Z_0+C_Z_alpha*alpha_trim+C_Z_de*d_e) - C_Z_alpha*w_trim/2. + u_trim*C_Z_q*P.c*q_trim/(4.*Va_trim))
        Zw = P.rho*P.S_wing/P.mass*(w_trim*(C_Z_0+C_Z_alpha*alpha_trim+C_Z_de*d_e) + C_Z_alpha*u_trim/2. + w_trim*P.c*C_Z_q*q_trim/(4*Va_trim))
        Zq = u_trim + P.rho*Va_trim*P.S_wing*C_Z_q*P.c/(4.*P.mass)
        Zde = P.rho*Va_trim**2*P.S_wing*C_Z_de/(2.*P.mass)

        Mu = P.rho*P.S_wing*P.c/P.jy*(u_trim*(P.C_m_0+P.C_m_alpha*alpha_trim+P.C_m_delta_e*d_e) - P.C_m_alpha*w_trim/2. + P.c*P.C_m_q*q_trim*u_trim/(4.*Va_trim))
        Mw = P.rho*P.S_wing*P.c/P.jy*(w_trim*(P.C_m_0+P.C_m_alpha*alpha_trim+P.C_m_delta_e*d_e) + P.C_m_alpha*u_trim/2. + P.c*P.C_m_q*q_trim*w_trim/(4.*Va_trim))
        Mq = P.rho*Va_trim*P.S_wing*P.c**2*P.C_m_q/(4.*P.jy)
        Mde = P.rho*Va_trim**2*P.S_wing*P.c*P.C_m_delta_e/(2.*P.jy)

        # Latxdot = [vdot, pdot, rdot, phidot, psidot]
        A = np.array([[Xu, Xw, Xq, -P.gravity*np.cos(theta_trim), 0.],
                    [Zu, Zw, Zq, -P.gravity*np.sin(theta_trim), 0.],
                    [Mu, Mw, Mq, 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [np.sin(theta_trim), -np.cos(theta_trim), 0., u_trim*np.cos(theta_trim) + w_trim*np.sin(theta_trim), 0.]])

        B = np.array([[Xde, Xdt],
                    [Zde, 0.],
                    [Mde, 0.],
                    [0., 0.],
                    [0., 0.]])
        
        x = np.array([[u_trim], [w_trim], [q_trim], [theta_trim], [h_trim]])
        u = np.array([[d_e], [d_t]])
        Longxdot = np.matmul(A, x) + np.matmul(B, u)
        
        return Longxdot, A, B
    
    def Modes(Alat, Alon):
        drcnt = 0; Po1 = np.zeros(2); Po2 = [] #Po2 = np.zeros(2) 
        j=0; cont1=True; cont2 = True

        LaVal = np.linalg.eig(Alat)[0]
        for i in range(0,len(LaVal)):
            if np.imag(LaVal[i]) != 0:
                drcnt += 1
                print("Dutch Roll Pole No.", drcnt, "is", LaVal[i])
            elif np.imag(LaVal[i]) == 0 and np.real(LaVal[i]) > 0:
                print("The Spiral-Divergence Pole is", LaVal[i])
            elif np.imag(LaVal[i]) == 0 and np.real(LaVal[i]) < 0:
                print("The Roll Mode Pole is", LaVal[i])

        LoVal= np.linalg.eig(Alon)[0]
        for i in range(0,len(LoVal)):
            while j < len(LoVal) and cont1==True:
                if j != i:
                    if np.real(LoVal[i]) == np.real(LoVal[j]):
                        Po1 = [LoVal[i], LoVal[j]]
                        cont1=False
                j += 1


            j = 0
            while j < len(LoVal) and cont2==True:
                if j != i and LoVal[j] != Po1[1]:
                    if np.real(LoVal[j]) == np.real(LoVal[i]):
                        Po2 = np.array([LoVal[j], LoVal[i]])
                        cont2 = False
                j += 1

        if np.sqrt(np.real(Po1[0])**2 + np.imag(Po1[0])) < np.sqrt(np.real(Po2[0])**2+np.imag(Po2[0])**2):
            print("Phugoid Pole No. 1 is:", Po1[0], "\nPhugoid Pole No. 2 is:", Po1[1])
            print("Short-Period Pole No. 1 is:", Po2[0], "\nShort-Period Pole No. 2 is:", Po2[1])
        else:
            print("Phugoid Pole No. 1 is: ", Po2[0], "\nPhugoid Pole No. 2 is:", Po2[1])
            print("Short-Period Pole No. 1 is:", Po1[0], "\nShort-Period Pole No. 2 is:", Po1[1])
        
        return LaVal, LoVal