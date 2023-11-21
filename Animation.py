import sys
sys.path.append('.')# one directory up
from math import cos, sin
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from tools.rotations import Quaternion2Euler, Quaternion2Rotation, Euler2Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tools.transformations as Tran
import parameters.simulation_parameters as P

class Animation():
    def __init__(self, miss_state0, plane_state0, scale=1):
        
        self.scale=scale
        self.flag_init = True
        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([miss_state0[1][0]-P.plot_lim,     miss_state0[1][0]+P.plot_lim])
        self.ax.set_ylim([miss_state0[0][0]-P.plot_lim,     miss_state0[0][0]+P.plot_lim])
        self.ax.set_zlim([-miss_state0[2][0]-P.plot_lim,    -miss_state0[2][0]+P.plot_lim])
        self.ax.set_title('3D Animation')
        self.ax.set_xlabel('East(m)')
        self.ax.set_ylabel('North(m)')
        self.ax.set_zlabel('Height(m)')
        
        self.update(miss_state0, plane_state0)

    # Target Plane Animation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plane_vertices(self,pn,pe,pd,phi,theta,psi):
        # Plane dimensional parameters
        fuse_l1 =2.8
        fuse_l2 =0.1-2.5
        fuse_l3 =5
        fuse_w =1.25
        fuse_h =1.0
        wing_l = 1.5
        wing_w = 5
        tail_l =1
        tail_h =0.8
        tailwing_w =2.5
        tailwing_l =0.75
        prop_l = 0
        prop_w = 0

        # Coordinates of each point on the plane
        V = np.array([[fuse_l1, 0, 0],
        [fuse_l2, fuse_w/2, -fuse_h/2],
        [fuse_l2, -fuse_w/2, -fuse_h/2],
        [fuse_l2, -fuse_w/2, fuse_h/2],
        [fuse_l2, fuse_w/2, fuse_h/2],
        [-fuse_l3-0.65, 0, 0],
        [-1.2-2, wing_w/2, 0],                          # wing
        [-wing_l-2, wing_w/2, 0],                       # wing
        [-wing_l-2, -wing_w/2, 0],                      # wing
        [-1.2-2, -wing_w/2, 0],                         # wing
        [-fuse_l3+tailwing_l-1.3, (tailwing_w/2), 0],   # horizontal tail
        [-fuse_l3-1.0, tailwing_w/2, 0],                # horizontal tail
        [-fuse_l3-1.0, -tailwing_w/2, 0],               # horizontal tail
        [-fuse_l3+tailwing_l-1.3, -tailwing_w/2, 0],    # horizontal tail
        [-fuse_l3+tailwing_l+1, 0, 0],                  # vertical tail
        [-fuse_l3+(tailwing_l/2)-0.5, 0, -tail_h],      # vertical tail
        [-fuse_l3-1.0, 0, -tail_h],                     # vertical tail
        [fuse_l1, prop_l/2, prop_w/2],                  # prop
        [fuse_l1, prop_l/2, -prop_w/2],                 # prop
        [fuse_l1, -prop_l/2, -prop_w/2],                # prop
        [fuse_l1, -prop_l/2, prop_w/2],                 # prop
        [0, 0, 0],                                      # wing
        [-wing_l-2.3, 0, 0],                            # wing
        [-fuse_l3-0.65, 0, 0],                          # horizontal tail
        [-fuse_l3+tailwing_l-0.25, 0, 0]])*self.scale   # horizontal tail
        pos_ned=np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep= np.tile(pos_ned, (25,1)) # 21 vertices
        R=Euler2Rotation(phi,theta,psi)

        # Rotate and Translate 
        vr=np.matmul(R,V.T).T
        vr=vr+ned_rep

        # Rotate for plotting north=y east=x h=-z
        R_plot=np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1]])
        
        vr=np.matmul(R_plot,vr.T).T

        Vl=vr.tolist()
        Vl1=Vl[0]
        Vl2=Vl[1]
        Vl3=Vl[2]
        Vl4=Vl[3]
        Vl5=Vl[4]
        Vl6=Vl[5]
        Vl7=Vl[6]
        Vl8=Vl[7]
        Vl9=Vl[8]
        Vl10=Vl[9]
        Vl11=Vl[10]
        Vl12=Vl[11]
        Vl13=Vl[12]
        Vl14=Vl[13]
        Vl15=Vl[14]
        Vl16=Vl[15]
        Vl17=Vl[16]
        Vl18=Vl[17]
        Vl19=Vl[18]
        Vl20=Vl[19]
        Vl21=Vl[20]
        Vl22=Vl[21]
        Vl23=Vl[22]
        Vl24=Vl[23]
        Vl25=Vl[24]

        verts=[[Vl1,Vl18,Vl19,Vl1],  
        [Vl1,Vl20,Vl21,Vl1],  
        [Vl1,Vl3,Vl4,Vl1],
        [Vl1,Vl2,Vl3,Vl1],
        [Vl1,Vl2,Vl5,Vl1],
        [Vl1,Vl4,Vl5,Vl1],
        [Vl3,Vl6,Vl4,Vl3],
        [Vl2,Vl6,Vl3,Vl2],
        [Vl2,Vl6,Vl5,Vl2],
        [Vl4,Vl5,Vl6,Vl4],
        [Vl7,Vl8,Vl23,Vl22],
        [Vl22,Vl23,Vl9,Vl10],
        [Vl11,Vl12,Vl24,Vl25],
        [Vl25,Vl24,Vl13,Vl14],
        [Vl6,Vl15,Vl16,Vl17]]  
        return(verts)    

    def draw_plane(self, pn, pe, pd, phi, theta, psi):
        verts=self.plane_vertices(pn,pe,pd,phi,theta,psi)
        if self.flag_init is True:
            poly = Poly3DCollection(verts, facecolors=['r', 'r', 'r', 'r', 'r','r'], alpha=.6)
            self.plane =self.ax.add_collection3d(poly)
        else:
            self.plane.set_verts(verts)

    # Missile Animation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def miss_vertices(self,pn,pe,pd,phi,theta,psi):
        # Missile dimensional parameters
        fuse_l1 = 1.9035   # CG to nose tip
        fuse_l2 = 1.5044   # CG to front of fuse
        fuse_l3 = 1.7965   # CG to back of missile
        fuse_d = 0.203     # Diameter of missile
        fin_l = 0.5      # Length of fins
        fin_h = 0.34        # Height of fins

        # Coordinates of each vertex on the missile
        Nose_tip = [fuse_l1, 0., 0.]
        Fuse_ttr = [fuse_l2, fuse_d/2, -fuse_d/2]
        Fuse_ttl = [fuse_l2, -fuse_d/2, -fuse_d/2]
        Fuse_tbr = [fuse_l2, fuse_d/2, fuse_d/2]
        Fuse_tbl = [fuse_l2, -fuse_d/2, fuse_d/2]

        Fuse_btr = [-fuse_l3, fuse_d/2, -fuse_d/2]
        Fuse_btl = [-fuse_l3, -fuse_d/2, -fuse_d/2]
        Fuse_bbr = [-fuse_l3, fuse_d/2, fuse_d/2]
        Fuse_bbl = [-fuse_l3, -fuse_d/2, fuse_d/2]        

        Fin1_t = [-fuse_l3, 0., fin_h]                  # xyz for top fin tip
        Fin1_r1 = [-(fuse_l3 - fin_l), 0., fuse_d/2]    # xyz for top fin leading root
        Fin1_r2 = [-fuse_l3, 0., fuse_d/2]              # xyz for top fin trailing root

        Fin2_t = [-fuse_l3, fin_h, 0.]
        Fin2_r1 = [-(fuse_l3 - fin_l), fuse_d/2, 0.]
        Fin2_r2 = [-fuse_l3, fuse_d/2, 0.]
        
        Fin3_t = [-fuse_l3, 0., -fin_h]
        Fin3_r1 = [-(fuse_l3 - fin_l), 0., -fuse_d/2]
        Fin3_r2 = [-fuse_l3, 0., -fuse_d/2]
        
        Fin4_t = [-fuse_l3, -fin_h, 0.]
        Fin4_r1 = [-(fuse_l3 - fin_l), -fuse_d/2, 0.]
        Fin4_r2 = [-fuse_l3, -fuse_d/2, 0.]

        # Creating an array of the verticies
        Missile=np.array([Nose_tip, Fuse_ttr, Fuse_ttl, Fuse_tbr, Fuse_tbl,
                        Fuse_btr, Fuse_btl, Fuse_bbr, Fuse_bbl, 
                        Fin1_r1, Fin1_r2, Fin1_t, 
                        Fin2_r1, Fin2_r2, Fin2_t, 
                        Fin3_r1, Fin3_r2, Fin3_t, 
                        Fin4_r1, Fin4_r2, Fin4_t])

        # Creating the starting position
        pos_ned=np.array([pn, pe, pd])

        # create m by n copies of pos_ned and used for translation
        ned_rep= np.tile(pos_ned, (21,1))

        # Create rotation matrix
        R=Euler2Rotation(phi,theta,psi)

        # Rotate then Translate plane verticies/vertex positions according to Euler angles
        vr=np.matmul(R,Missile.T).T
        vr=vr+ned_rep

        # Rotate for plotting north=y east=x h=-z
        R_plot=np.array([[0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]])
            
        vr=np.matmul(R_plot,vr.T).T
        Vl=vr.tolist()
        V11=Vl[0] # Nose Tip
        V12=Vl[1] # Fuse tip top right
        V13=Vl[2] # Fuse tip top left
        V14=Vl[3] # Fuse tip bottom right
        V15=Vl[4] # Fuse tip bottom left
        
        V21=Vl[5] # Fuse bot top right
        V22=Vl[6] # Fuse bot top left
        V23=Vl[7] # Fuse bot bottom right
        V24=Vl[8] # Fuse bot bottom left

        V31=Vl[9]  # Top fin leading edge root
        V32=Vl[10] # Top fin trailing edge root
        V33=Vl[11] # Top fin tip

        V41=Vl[12] # Right fin leading edge root
        V42=Vl[13] # Right fin trailing edge root
        V43=Vl[14] # Right fin tip

        V51=Vl[15] # Bot fin leading edge root
        V52=Vl[16] # Bot fin trailing edge root
        V53=Vl[17] # Bot fin tip

        V61=Vl[18] # Left fin leading edge root
        V62=Vl[19] # Left fin trailing edge root
        V63=Vl[20] # Left fin tip

        verts=[[V11,V12,V13],  # Fuse face 1 (top nose)
        [V11,V12,V14], # Fuse face 2 (right nose)
        [V11,V13,V15], # Fuse face 3 (left nose)
        [V11,V14,V15], # Fuse face 4 (bottom nose)
        [V12,V13,V22,V21], # Fuse face 5 (top fuse)
        [V12,V14,V23,V21], # Fuse face 6 (right fuse)
        [V13,V15,V24,V22], # Fuse face 7 (left fuse)
        [V14,V15,V24,V23], # Fuse Face 8 (bottom fuse)
        [V21,V22,V24,V23], # Fuse Face 9 (back fuse)
        [V31,V32,V33], # Top Fin
        [V41,V42,V43], # Right Fin
        [V51,V52,V53], # Bottom Fin
        [V61,V62,V63]] # Left Fin
        return(verts)    

    # Update missile (convert to update plane + missile?)
    def update(self, mis_state, plane_state):
        mis_pn = mis_state[0][0];       pl_pn = plane_state[0][0]
        mis_pe = mis_state[1][0];       pl_pe = plane_state[1][0]
        mis_pd = mis_state[2][0];       pl_pd = plane_state[2][0]
        mis_phi = mis_state[6][0];      pl_phi = plane_state[6][0]
        mis_theta = mis_state[7][0];    pl_theta = plane_state[7][0]
        mis_psi = mis_state[8][0];      pl_psi = plane_state[8][0]
    
        # Draw plot elements: Missile & Plane
        self.draw_missile(mis_pn,mis_pe,mis_pd,mis_phi,mis_theta,mis_psi)
        self.draw_plane(pl_pn, pl_pe, pl_pd, pl_phi, pl_theta, pl_psi)

        # Set initialization flag to False after first call
        if self.flag_init == True:
            self.flag_init = False

        # Keep the missile at the center of the plot
        self.ax.set_xlim([mis_pe-P.plot_lim,    mis_pe+P.plot_lim])
        self.ax.set_ylim([mis_pn-P.plot_lim,    mis_pn+P.plot_lim])
        self.ax.set_zlim([-mis_pd-P.plot_lim,   -mis_pd+P.plot_lim])

    def draw_missile(self, pn, pe, pd, phi, theta, psi):
        verts=self.miss_vertices(pn,pe,pd,phi,theta,psi)
        if self.flag_init is True:
            poly = Poly3DCollection(verts, facecolors=['b', 'b', 'b', 'b', 'b','b'], alpha=.6)
            self.miss = self.ax.add_collection3d(poly)# 
        else:
            self.miss.set_verts(verts)