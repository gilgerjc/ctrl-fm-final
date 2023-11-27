import sys
sys.path.append('..')
import numpy as np

gravity = 9.806650

######################################################################################
                #   Missile Params
######################################################################################

# all in kg*m^2
# Jxx = 1.957; Jyy = 2.677 E 2; Jzz = 2.677 E 2; Jxz = 0.025 E -9 
mi_jx= 1.957;  mi_jy= 267.7;  mi_jz= 267.7;  mi_jxz= 0.25*10**-9
mi_mass = 246.3

mi_G = mi_jx*mi_jz-mi_jxz**2
mi_G1 = (mi_jxz*(mi_jx-mi_jy+mi_jz))/mi_G;    mi_G2 = (mi_jz*(mi_jz-mi_jy)+mi_jxz**2)/mi_G;    mi_G3 = mi_jz/mi_G
mi_G4 = mi_jxz/mi_G;                          mi_G5 = (mi_jz-mi_jx)/mi_jy;                     mi_G6 = mi_jxz/mi_jy
mi_G7 = ((mi_jx-mi_jy)*mi_jx+mi_jxz**2)/mi_G; mi_G8 = mi_jx/mi_G


# Initial States
mis_states0=np.array([[0.], [0.], [-100.], [50.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])


## Aerodynamic parameters
# Planform: 0.0419638 m^2; LE: 0.005294175 m^2; TE: 0.00582286 m^2
mi_S = 0.366713;  mi_b = 0.94;   mi_c = 0.641667;   mi_AR = mi_b**2/mi_S
rho = 1.2682;   e = 0.9

M = 50.
alpha0 = 0.47
epsilon = 0.16

mi_k_th = 35000.
k_Tp = 0.
k_omega = 0.

mi_C_L_0 = 0.0;       mi_C_D_0 = 0.043;          mi_C_m_0 = 0.0135
mi_C_L_alpha = np.pi*mi_AR/(1+np.sqrt(1+(mi_AR/2)**2));   mi_C_D_alpha = 0.030;      mi_C_m_alpha = -2.74
mi_C_L_q = 4.92962;       mi_C_D_q = 0.0;            mi_C_m_q = -0.64641
mi_C_L_delta_e = 0.11111; mi_C_D_delta_e = 0.011539;   mi_C_m_delta_e = -6.47839

M = 50
alpha0 = 0.47
epsilon = 0.16

mi_C_D_p = 0.0
mi_C_Y_0 = 0.0;    mi_C_Y_beta = -0.83765;   mi_C_Y_p = 0.0;            mi_C_Y_r = 0.0;    mi_C_Y_delta_a = 0.075;  mi_C_Y_delta_r = 0.19
mi_C_ell_0 = 0.0;  mi_C_ell_beta = -0.044625; mi_C_ell_delta_a = 0.1167
mi_C_n_0 = 0.0;    mi_C_n_beta = 1.60576;   mi_C_n_r = -3.05992;         mi_C_n_p = -3.5923; mi_C_n_delta_a = -0.241964; mi_C_n_delta_r = -3.03555
mi_C_ell_p = -3.5287; mi_C_ell_r = 2.5946;     mi_C_ell_delta_r = 0.0016477

C_p_0 = mi_G3*mi_C_ell_0 + mi_G4*mi_C_n_0
C_p_beta = mi_G3*mi_C_ell_beta + mi_G4*mi_C_n_beta
C_p_p = mi_G3*mi_C_ell_p + mi_G4*mi_C_n_p
C_p_r = mi_G3*mi_C_ell_r + mi_G4*mi_C_n_r
C_p_da = mi_G3*mi_C_ell_delta_a + mi_G4*mi_C_n_delta_a
C_p_dr = mi_G3*mi_C_ell_delta_r + mi_G4*mi_C_n_delta_r
C_r_0 = mi_G4*mi_C_ell_0 + mi_G8*mi_C_n_0
C_r_beta = mi_G4*mi_C_ell_beta + mi_G8*mi_C_n_beta
C_r_p = mi_G4*mi_C_ell_p + mi_G8*mi_C_n_p
C_r_r = mi_G4*mi_C_ell_r + mi_G8*mi_C_n_r
C_r_da = mi_G4*mi_C_ell_delta_a + mi_G8*mi_C_n_delta_a
C_r_dr = mi_G4*mi_C_ell_delta_r + mi_G8*mi_C_n_delta_r

######################################################################################
                #   Plane Params
######################################################################################
# Plane Inertial parameters
# All in kg*m^2; Apply scale factor of 0.45044 after conversion (given for a model of mass 19370 kg)
# Jxx = 2.082 E 4 --> 9378.136; Jyy = 1.645 E 5 --> 74097.186; Jzz = 1.771 E 5 --> 79772.716; Jxz = -6.235 E 3 --> 2808.486
# (Jxy = 1.245 --> 0.5608; Jyz = -1.303 --> 0.5869)
pl_jx= 9378.136;  pl_jy= 74097.186;  pl_jz= 79772.716;  pl_jxz= 2808.486
pl_mass = 8725.

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
pl_S = 247.5;  pl_b = 7.16;   pl_c = 5.97;   pl_AR = pl_b**2/pl_S
rho = 1.2682;   e = 0.9

pl_k_th = 5748.
k_Tp = 0.
k_omega = 0.

pl_C_L_0 = 0.15;       pl_C_D_0 = 0.043;          pl_C_m_0 = 0.0135
pl_C_L_alpha = np.pi*pl_AR/(1+np.sqrt(1+(pl_AR/2)**2));   pl_C_D_alpha = 0.030;      pl_C_m_alpha = -2.74
pl_C_L_q = 0.0452619;       pl_C_D_q = 0.0;            pl_C_m_q = -2.7974
pl_C_L_delta_e = 0.013242; pl_C_D_delta_e = 0.001375;   pl_C_m_delta_e = -0.64837

M = 50
alpha0 = 0.47
epsilon = 0.16

pl_C_D_p = 0.0
pl_C_Y_0 = 0.0;    pl_C_Y_beta = -0.09982;   pl_C_Y_p = 0.0;            pl_C_Y_r = 0.0;    pl_C_Y_delta_a = 0.075;  pl_C_Y_delta_r = 0.19
pl_C_ell_0 = 0.0;  pl_C_ell_beta = -0.09445; pl_C_ell_delta_a = 0.247
pl_C_n_0 = 0.0;    pl_C_n_beta = 0.21133;   pl_C_n_r = -0.782522;         pl_C_n_p = -1.1367; pl_C_n_delta_a = -0.031845; pl_C_n_delta_r = -0.3995085
pl_C_ell_p = -2.1085; pl_C_ell_r = 0.51679;     pl_C_ell_delta_r = 19.8393

pl_C_p_0 = pl_G3*pl_C_ell_0 + pl_G4*pl_C_n_0
pl_C_p_beta = pl_G3*pl_C_ell_beta + pl_G4*pl_C_n_beta
pl_C_p_p = pl_G3*pl_C_ell_p + pl_G4*pl_C_n_p
pl_C_p_r = pl_G3*pl_C_ell_r + pl_G4*pl_C_n_r
pl_C_p_da = pl_G3*pl_C_ell_delta_a + pl_G4*pl_C_n_delta_a
pl_C_p_dr = pl_G3*pl_C_ell_delta_r + pl_G4*pl_C_n_delta_r
pl_C_r_0 = pl_G4*pl_C_ell_0 + pl_G8*pl_C_n_0
pl_C_r_beta = pl_G4*pl_C_ell_beta + pl_G8*pl_C_n_beta
pl_C_r_p = pl_G4*pl_C_ell_p + pl_G8*pl_C_n_p
pl_C_r_r = pl_G4*pl_C_ell_r + pl_G8*pl_C_n_r
pl_C_r_da = pl_G4*pl_C_ell_delta_a + pl_G8*pl_C_n_delta_a
pl_C_r_dr = pl_G4*pl_C_ell_delta_r + pl_G8*pl_C_n_delta_r 

######################################################################################
                #   Simulation Params
######################################################################################
plot_lim = 15.

ts_simulation = 0.001    # smallest time step for simulation
start_time = 0.         # start time for simulation
end_time = 40.          # end time for simulation
ts_plotting = 0.01      # refresh rate for plots

ts_sensor = 0.05  # sample rate for the controller