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
from viewers.dataPlotter import dataPlotter
import dynamics.Plane_compute_trim as Pl_CompT
import dynamics.Missile_compute_trim as Mi_CompT
import dynamics.autopilot as Auto

# Create trim condition for plane
Vatgt = 250.; Ytgt = 0.; Rtgt = np.inf
CT = Pl_CompT.ComputeTrim()
pl_x_trim, pl_d = CT.compute_trim(Vatgt, Ytgt, Rtgt)
pl_x_trim[0][0] = 50; pl_x_trim[1][0] = -30.; pl_x_trim[2][0] = -90.

# Create trim states for missile for gains calculations
Vatgt = 200.
mi_CT = Mi_CompT.ComputeTrim()
mi_x_trim, mi_d_trim = mi_CT.compute_trim(Vatgt, Ytgt, Rtgt)
mi_x_trim[0][0] = 0.; mi_x_trim[1][0] = 0.; mi_x_trim[2][0] = -100.
k_phi, k_chi, k_thet = Mi_CompT.compute_gains.compute_tf_models(mi_x_trim, mi_d_trim)

# Initialize Animation, Dynamics, Data Plotter, Input Methods
anim =  Animation(P.mis_states0, pl_x_trim)
pl_dyn = Plane.UAVDynamics(pl_x_trim)                           # MiG-21
mi_dyn = Missile.MisDynamics(P.mis_states0)                     # AIM-7A Missile
dp = dataPlotter()

# Initialize the simulation
sim_time = P.start_time
ts_plot = P.ts_plotting
ts_radar = P.start_time
print("Press Command-Q to exit...")

# Create PN Stuff
Ts = P.ts_simulation
Thet_c = 0.; ttd_l = 0.
Psi_c = 0.; psd_l = 0.
kPN_la = 5.
kPN_lo = 5.
# dist_l = 0.
# sph_thet_l = 0.
# sph_phi_l = 0.

# Create array of PID gains
k = np.array([[1.*k_phi[0], 0.04*k_phi[1], 0.],     # Kp, Kd, Ki for roll control
              [2.7*k_thet[0], 0.9*k_thet[1], 1.*0.2],     # Kp, Kd, Ki for pitch control
              [-1.3*k_chi[0], 0., 0.8*k_chi[2]]])    # Kp, Kd, Ki for yaw control

while sim_time < P.end_time:

    if sim_time >= ts_radar:
        # Radar collects distance and angles from target at radar sampling rate. Values calculated via spherical coordinates
        x_dist = pl_dyn.state[0][0] - mi_dyn.state[0][0]                    # Distance from missile to plane in x in Vehicle Frame 
        y_dist = pl_dyn.state[1][0] - mi_dyn.state[1][0]                    # Distance from missile to plane in y in Vehicle Frame
        z_dist = pl_dyn.state[2][0] - mi_dyn.state[2][0]                    # Distance from missile to plane in z in Vehicle Frame

        dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)                   # Distance from missile to plane in spherical coordinates based on Vehicle Frame
        sph_thet = np.arctan(y_dist/x_dist)                               # Distance's angle from x-axis in spherical coordinates based on Vehicle Frame
        sph_phi = np.arctan( -z_dist/np.sqrt(x_dist**2 + y_dist**2) )      # Distance's angle from projection of dist on xy-plane in spherical coordinates, based on Vehicle Frame

        N_c = dist*np.cos(sph_phi)*np.cos(sph_thet) + mi_dyn.state[0][0]    # Target x-position in vehicle for plotting
        E_c = dist*np.cos(sph_phi)*np.sin(sph_thet) + mi_dyn.state[1][0]    # Target y-position in vehicle for plotting
        D_c = dist*np.sin(sph_phi) + mi_dyn.state[2][0]                     # Target z-position in vehicle for plotting

        # Implement PN Guidance (state estimation)
        # Psidot_c = Missile.Guidance.Lat_PN(dist,dist_l,sph_thet,sph_thet_l,sph_phi,sph_phi_l,kPN_la)
        # Thetadot_c = Missile.Guidance.Lon_PN(dist,dist_l,sph_thet,sph_thet_l,sph_phi,sph_phi_l,kPN_lo)
        # Phi_c = 0.

        # Implement PN Guidance Proof of Concept (known states)
        # Thetadot_c, Psidot_c = Missile.Guidance.PN_Demo(mi_dyn.state,pl_dyn.state,kPN_lo,kPN_la)

        # Update last values for next iteration (state estimation)
        # dist_l = dist
        # sph_thet_l = sph_thet
        # sph_phi_l = sph_phi

        if dist <= 7.6:
            print("Hit")

        ts_radar += P.ts_sensor

    ts_plot += P.ts_plotting
    while sim_time < ts_plot:

        # Thet_c += 1/2*(Ts)*(Thetadot_c+ttd_l)       # Estimate theta for PID control
        # Psi_c += 1/2*(Ts)*(Psidot_c+psd_l)          # Estimate psi for PID control

        # Find deflections from autopilot/state machine
        d_a, d_e, d_r = Auto.autopilot(sim_time, mi_dyn.state[6][0], sph_phi, mi_dyn.state[7][0], sph_thet, mi_dyn.state[8][0], k)
        mi_d = np.array([[d_a], [d_e], [d_r]])                      # Array of missile deflections
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
        # mi_dyn.state[6][0] = 0.; mi_dyn.state[9][0] = 0.

        # if state == 1:
        #     print("Distance:",dist," Spherical Theta:",sph_thet," Spherical Phi:",sph_phi, " Mode: Roll")
        # elif state == 2:
        #     print("Distance:",dist," Spherical Theta:",sph_thet," Spherical Phi:",sph_phi, " Mode: yaw")

        sim_time += P.ts_simulation

    tgts = np.array([N_c, E_c, D_c, sph_phi, sph_thet])
    dp.update(sim_time, mi_dyn.state, pl_dyn.state, mi_d, tgts)                        # Update plot including target values
    anim.update(mi_dyn.state, pl_dyn.state) # -pd for height                # Update Plane and Missile in animation
    plt.pause(0.0001)
