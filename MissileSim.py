# Import Python Libraries
import sys
sys.path.append('C:/Users/cmack/OneDrive - University of Cincinnati/Desktop/Flight Mechanics/Python_Files/Final_Project')# one directory up
import numpy as np
import matplotlib.pyplot as plt
import parameters.simulation_parameters as P

# Import Sharma Modules
# from tools.sliders import sliders
# from tools.signalGenerator import signalGenerator

# Import my Modules
import dynamics.Plane_Dynamics as Plane
import dynamics.Missile_Dynamics as Missile
from viewers.Animation import Animation
# from viewers.dataPlotter import dataPlotter
import dynamics.compute_trim as CompT
# import dynamics.autopilot as Auto

Vatgt = 100.; Ytgt = 0.; Rtgt = np.inf
CT = CompT.ComputeTrim()
pl_x_trim, pl_d = CT.compute_trim(Vatgt, Ytgt, Rtgt)
pl_x_trim[2][0] = -90.

# Initialize Animation, Dynamics, Data Plotter, Input Methods
anim =  Animation(P.mis_states0, pl_x_trim)
pl_dyn = Plane.UAVDynamics(pl_x_trim)                           # MiG ...?
mi_dyn = Missile.MisDynamics(P.mis_states0)                     # AIM-7A Missile
# dp =        dataPlotter()

# Initialize the simulation
sim_time = P.start_time
ts_plot = P.ts_plotting
ts_radar = P.start_time
print("Press Command-Q to exit...")

while sim_time < P.end_time:

    # if sim_time >= ts_radar:
        #   Radar collects distance and angles from target at radar sampling rate. Values calculated via spherical coordinates
        #   dist = np.sqrt((pl_dyn.state[0][0]-mi_dyn.state[0][0])**2 + (pl_dyn.state[1][0]-mi_dyn.state[1][0])**2 + (pl_dyn.state[2][0]-mi_dyn.state[2][0])**2)
        #   sph_thet = np.arctan2((pl_dyn.state[1][0]-mi_dyn.state[1][0]),(pl_dyn.state[0][0]-mi_dyn.state[0][0]))
        #   sph_phi = np.arctan2((np.sqrt((pl_dyn.state[0][0]-mi_dyn.state[0][0])**2 + (pl_dyn.state[1][0]-mi_dyn.state[1][0])**2)),pl_dyn.state[2][0]-mi_dyn.state[2][0])
        #   Tgt = np.array([dist, sph_thet, sph_phi])  

        #   ts_radar += P.ts_sensor

    ts_plot += P.ts_plotting
    while sim_time < ts_plot:

        # Missile Control and Simulation
        # Thrust as a func of time of the form T = at*b^-ct
        # d_1, d_2, d_3, d_4 = mi_dyn.PID(Ntgt, Etgt, Dtgt, Kp, Kd, Ki)          # Missile deflections from PID ctrl
        mi_d = np.array([[-5.*np.pi/180.], [0.], [5.*np.pi/180.], [0.]])
        fx, fy, fz = Missile.forces_moments.forces(mi_dyn.state, mi_d)     # Missile forces from deflections
        L, M, N = Missile.forces_moments.moments(mi_dyn.state, mi_d)       # Missile moments from deflections
        miU = np.array([[fx], [fy], [fz], [L], [M], [N]])              # Missile dynamics input at sim_time

        # Target Plane Control and Simulation
        fx, fy, fz, = Plane.forces_moments.forces(pl_dyn.state, pl_d)       # Plane forces from deflections
        L, M, N = Plane.forces_moments.moments(pl_dyn.state, pl_d)          # Plane moments from deflections
        plU = np.array([[fx], [fy], [fz], [L], [M], [N]])              # Plane dynamics input at sim_time

        # Update Plane and Missile States using above forces & moments
        pl_dyn.update(plU)
        mi_dyn.update(miU)

        sim_time += P.ts_simulation

    # dp.update(sim_time, dyn.state, d, VaMag, tgts)                        # Update plot including target values
    anim.update(mi_dyn.state, pl_dyn.state) # -pd for height                # Update Plane and Missile in animation
    plt.pause(0.0001)
