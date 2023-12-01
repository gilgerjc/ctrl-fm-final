import numpy as np
import dynamics.hold_loops as LP
import dynamics.Missile_Dynamics as MD
import parameters.simulation_parameters as P

def autopilot(t, phi, thetadot_c, thetadot, psidot_c, psidot, k):
    '''d_1, d_2, d_3, d_4, state = autopilot(u)
    Takes arguments of the current simulation time, the commanded roll angle, the current roll angle,
    the commanded pitch rate, the current pitch rate, the commanded yaw rate, the current yaw rate,
    and a 3x3 matrix of gains. (4 current-state, 3 commanded values, 1 array of gains).

    Calculates, with consideration taken to phase of flight (prioritize roll, prioritize yaw), the required
    deflections of the right elevator (d_1), the bottom rudder (d_2), the left elevator (d_3), the top rudder (d_4), 
    required roll angle, required pitch angle, and required course angle.
    
    Returns all calculated values, all commanded values, and the aircraft's current state (roll or yaw). 
    '''

    # Set variables that will retain their values through different iterations
    global dt
    global reset_yaw
    global reset_roll
    global reset_pitch 

    # Pull variables from the simulation parameters
    roll_hold_range = P.roll_hold_range
    dt = P.ts_simulation

    # Separate out gains from input gain matrix
    k_roll = k[0]
    k_pit = k[1]
    k_yaw = k[2]

    # Set integrators and differentiators to zero when the simulation first starts
    if t == 0:
        reset_pitch = True
        reset_roll = True
        reset_yaw = True

    # State Machine
    # if phi < phi_c - roll_hold_range or phi > phi_c + roll_hold_range:      # If missile is not at acceptable roll angle, roll state
    #     mode = 1                                                            # Indicate roll mode
    #     d_2, d_4 = MD.Guidance.roll_PID(phi_c, phi, k_roll, reset_roll)     # Determine bottom and top deflections per PID
    #     reset_yaw = True                                                    # Next time you leave roll mode, reset integ + diff for yaw
    #     reset_roll = False                                                  # Do not reset integ + diff for roll while in this mode

    # else:                                                                   # If missile is at acceptable roll angle, yaw state
    #     mode = 2                                                            # Indicate yaw mode
    #     d_2, d_4 = MD.Guidance.yaw_PID(psidot_c, psidot, k_yaw, reset_yaw)  # Determine bottom and top deflections per PID
    #     reset_roll = True                                                   # Next Time you leave yaw mode, reset integ + diff for roll
    #     reset_yaw = False                                                   # Do not reset integ + diff for yaw while in this mode

    phi_c = MD.Guidance.yaw_PID(psidot_c, psidot, k_yaw, reset_yaw)
    d_a = MD. Guidance.roll_PID(phi_c, phi, k_roll, reset_roll)
    d_e = MD.Guidance.pit_PID(thetadot_c, thetadot, k_pit, reset_pitch)    # Determine right and left deflections per PID
    d_r = 0.

    reset_roll = False
    reset_pitch = False                                                         # Do not reset integ + diff for pitch after initialization
    reset_yaw = False

    mode = 0
    return (d_a,d_e,d_r)
