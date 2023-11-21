import sys
sys.path.append('..')
import numpy as np

gravity = 9.806650

######################################################################################
                #   Missile Params
######################################################################################

mi_jx= 0.333;  mi_jy= 0.96;  mi_jz= 0.96;  mi_jxz= 0.25
mi_mass = 143.

mi_G = mi_jx*mi_jz-mi_jxz**2
mi_G1 = (mi_jxz*(mi_jx-mi_jy+mi_jz))/mi_G;    mi_G2 = (mi_jz*(mi_jz-mi_jy)+mi_jxz**2)/mi_G;    mi_G3 = mi_jz/mi_G
mi_G4 = mi_jxz/mi_G;                          mi_G5 = (mi_jz-mi_jx)/mi_jy;                     mi_G6 = mi_jxz/mi_jy
mi_G7 = ((mi_jx-mi_jy)*mi_jx+mi_jxz**2)/mi_G; mi_G8 = mi_jx/mi_G

# Initial States
mis_states0=np.array([[0.], [0.], [-100.], [45.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])

######################################################################################
                #   Plane Params
######################################################################################
# Plane Inertial parameters
pl_jx= 0.824;  pl_jy= 1.135;  pl_jz= 1.759;  pl_jxz= 0.25
pl_mass = 500.

pl_G = pl_jx*pl_jz-pl_jxz**2
pl_G1 = (pl_jxz*(pl_jx-pl_jy+pl_jz))/pl_G;      pl_G2 = (pl_jz*(pl_jz-pl_jy)+pl_jxz**2)/pl_G;      pl_G3 = pl_jz/pl_G
pl_G4 = pl_jxz/pl_G;                            pl_G5 = (pl_jz-pl_jx)/pl_jy;                       pl_G6 = pl_jxz/pl_jy
pl_G7 = ((pl_jx-pl_jy)*pl_jx+pl_jxz**2)/pl_G;   pl_G8 = pl_jx/pl_G

# inital states
pl_states0=np.array([[15.], [0.], [-95.], [50.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]) 

# Wind Parameters
Lu = 200.;  Lv = 200.;  Lw = 50.
Su = 1.06;  Sv = 1.06;  Sw = 0.7

## aerodynamic parameters
S_wing = 0.55;  b = 2.90;   c = 0.19;   AR = b**2/S_wing
rho = 1.2682;   e = 0.9

C_prop = 1.; S_prop = 0.2027
k_motor = 80. #80
k_Tp = 0.
k_omega = 0.

C_L_0 = 0.23;       C_D_0 = 0.043;          C_m_0 = 0.0135
C_L_alpha = 5.61;   C_D_alpha = 0.030;      C_m_alpha = -2.74
C_L_q = 7.95;       C_D_q = 0.0;            C_m_q = -38.21
C_L_delta_e = 0.13; C_D_delta_e = 0.0135;   C_m_delta_e = -0.99

M = 50
alpha0 = 0.47
epsilon = 0.16

C_D_p = 0.0
C_Y_0 = 0.0;    C_Y_beta = -0.98;   C_Y_p = 0.0;            C_Y_r = 0.0;    C_Y_delta_a = 0.075;  C_Y_delta_r = 0.19
C_ell_0 = 0.0;  C_ell_beta = -0.13; C_ell_delta_a = 0.17
C_n_0 = 0.0;    C_n_beta = 0.073;   C_n_r = -0.095;         C_n_p = -0.069; C_n_delta_a = -0.011; C_n_delta_r = -0.069
C_ell_p = -0.51;C_ell_r = 0.25;     C_ell_delta_r = 0.0024

C_p_0 = pl_G3*C_ell_0 + pl_G4*C_n_0
C_p_beta = pl_G3*C_ell_beta + pl_G4*C_n_beta
C_p_p = pl_G3*C_ell_p + pl_G4*C_n_p
C_p_r = pl_G3*C_ell_r + pl_G4*C_n_r
C_p_da = pl_G3*C_ell_delta_a + pl_G4*C_n_delta_a
C_p_dr = pl_G3*C_ell_delta_r + pl_G4*C_n_delta_r
C_r_0 = pl_G4*C_ell_0 + pl_G8*C_n_0
C_r_beta = pl_G4*C_ell_beta + pl_G8*C_n_beta
C_r_p = pl_G4*C_ell_p + pl_G8*C_n_p
C_r_r = pl_G4*C_ell_r + pl_G8*C_n_r
C_r_da = pl_G4*C_ell_delta_a + pl_G8*C_n_delta_a
C_r_dr = pl_G4*C_ell_delta_r + pl_G8*C_n_delta_r 

######################################################################################
                #   Simulation Params
######################################################################################
plot_lim = 2.

ts_simulation = 0.005    # smallest time step for simulation
start_time = 0.         # start time for simulation
end_time = 40.          # end time for simulation
ts_plotting = 0.01      # refresh rate for plots

ts_sensor = 0.05  # sample rate for the controller