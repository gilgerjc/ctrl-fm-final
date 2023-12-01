import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter:
    ''' 
        This class plots the time histories for the plane sim data.
    '''

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 6    # Number of subplot rows
        self.num_cols = 2    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True, constrained_layout=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.mi_pn_history = [];            self.pl_pn_history = []     # North Position data
        self.mi_pe_history = [];            self.pl_pe_history = []     # East Position data
        self.mi_pd_history = [];            self.pl_pd_history = []     # Down Position data
        self.mi_phi_history = [];           self.pl_phi_history = []    # Roll data
        self.mi_theta_history = [];         self.pl_theta_history = []  # Pitch data
        self.mi_psi_history = [];           self.pl_psi_history = []    # Yaw data 
        self.mi_q_history = []
        self.mi_r_history = []

        self.mi_VaMag_history = [];         self.pl_VaMag_history = []  # Airspeed data
        self.mi_d_a_history = [];         #self.pl_d_ail_history = []  # Aileron deflection data
        self.mi_d_e_history = [];         #self.pl_d_ele_history = []  # Elevator deflection data
        self.mi_d_r_history = [];         #self.pl_d_rud_history = []  # Rudder deflection data

        self.refpn_history = []
        self.refpe_history = []
        self.refpd_history = []
        # self.refphi_history = []
        self.reftheta_c_history = []
        self.refpsi_c_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0,0], ylabel='pn (m)', title='States'))
        self.handle.append(myPlot(self.ax[1,0], ylabel='pe (m)'))
        self.handle.append(myPlot(self.ax[2,0], ylabel='h (m)'))

        self.handle.append(myPlot(self.ax[3,0], ylabel='phi (deg)'))
        self.handle.append(myPlot(self.ax[4,0], ylabel='theta (deg)'))
        self.handle.append(myPlot(self.ax[5,0], xlabel = 'time (s)', ylabel='psi (deg)'))

        self.handle.append(myPlot(self.ax[0,1], ylabel='Va (m/s)', title="Trim & Inputs"))
        # self.handle.append(myPlot(self.ax[1,1], ylabel='psi_c (deg/s)'))

        # self.handle.append(myPlot(self.ax[2,1], ylabel='thet_c (deg/s)'))
        self.handle.append(myPlot(self.ax[3,1], ylabel='d_ail'))
        self.handle.append(myPlot(self.ax[4,1], ylabel='d_ele'))
        self.handle.append(myPlot(self.ax[5,1], xlabel='time (s)', ylabel='d_rud'))

    def update(self, t, mi_states, pl_states, mi_u, tgtVal):
        '''Add to the time and data histories, and update the plots.'''

        # Retrieve target values for plotting
        pn_c = tgtVal[0]
        pe_c = tgtVal[1]
        pd_c = tgtVal[2]
        thet_c = tgtVal[3]
        psi_c = tgtVal[4]
        # phi_c = tgtVal[3]
        # theta_c = tgtVal[4]
        # psi_c = tgtVal[5]

        # update the time history of all plot variables
        mi_pn = mi_states[0][0];    pl_pn = pl_states[0][0]
        mi_pe = mi_states[1][0];    pl_pe = pl_states[1][0]
        mi_pd = mi_states[2][0];    pl_pd = pl_states[2][0]
        mi_phi = mi_states[6][0];   pl_phi = pl_states[6][0]
        mi_theta = mi_states[7][0]; pl_theta = pl_states[7][0]
        mi_psi = mi_states[8][0];   pl_psi = pl_states[8][0]
        mi_q = mi_states[10][0]
        mi_r = mi_states[11][0]

        mi_Va = np.sqrt(mi_states[3][0]**2 + mi_states[4][0]**2 + mi_states[5][0]**2)
        pl_Va = np.sqrt(pl_states[3][0]**2 + pl_states[4][0]**2 + pl_states[5][0]**2)

        mi_d_a = mi_u[0][0];    #pl_d_a = pl_u[0][0]  
        mi_d_e = mi_u[1][0];    #pl_d_e = pl_u[1][0]
        mi_d_r = mi_u[2][0];    #pl_d_r = pl_u[2][0]
        # mi_d_t = mi_u[3][0];    pl_d_t = pl_u[3][0]

        self.time_history.append(t)  # time
        self.mi_theta_history.append(mi_theta*180./np.pi);      self.pl_theta_history.append(pl_theta*180./np.pi)   # pitch
        self.mi_phi_history.append(mi_phi*180./np.pi);          self.pl_phi_history.append(pl_phi*180./np.pi)       # roll
        self.mi_psi_history.append(mi_psi*180./np.pi);          self.pl_psi_history.append(pl_psi*180./np.pi)       # yaw
        self.mi_pn_history.append(mi_pn);                       self.pl_pn_history.append(pl_pn)                    # North Position
        self.mi_pe_history.append(mi_pe);                       self.pl_pe_history.append(pl_pe)                    # East Positiuon
        self.mi_pd_history.append(-mi_pd);                      self.pl_pd_history.append(-pl_pd)                   # Down Position
        self.mi_q_history.append(mi_q*180./np.pi)
        self.mi_r_history.append(mi_r*180./np.pi)

        self.mi_VaMag_history.append(mi_Va);                 self.pl_VaMag_history.append(pl_Va)              # Force in x
        # self.FPA_history.append(FPA*180./np.pi)  # 
        self.mi_d_a_history.append(mi_d_a);                     #self.pl_d_t_history.append(pl_d_t)                  # Force in z
        self.mi_d_e_history.append(mi_d_e);                   #self.pl_d_ail_history.append(pl_d_a)                # Moment about x-axis
        self.mi_d_r_history.append(mi_d_r);                   #self.pl_d_ele_history.append(pl_d_e)                # Moment about y-axis

        self.refpn_history.append(pn_c)
        self.refpe_history.append(pe_c)
        self.refpd_history.append(-pd_c)
        # self.refphi_history.append(phi_c*180./np.pi)
        self.reftheta_c_history.append(thet_c*180./np.pi)
        self.refpsi_c_history.append(psi_c*180./np.pi)

        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.mi_pn_history, self.pl_pn_history, self.refpn_history])
        self.handle[1].update(self.time_history, [self.mi_pe_history, self.pl_pe_history, self.refpe_history])
        self.handle[2].update(self.time_history, [self.mi_pd_history, self.pl_pd_history, self.refpd_history])

        self.handle[3].update(self.time_history, [self.mi_phi_history, self.pl_phi_history])
        self.handle[4].update(self.time_history, [self.mi_theta_history, self.pl_theta_history, self.reftheta_c_history])
        self.handle[5].update(self.time_history, [self.mi_psi_history, self.pl_psi_history, self.refpsi_c_history])

        self.handle[6].update(self.time_history, [self.mi_VaMag_history, self.pl_VaMag_history])
        # self.handle[7].update(self.time_history, [self.mi_r_history, self.refpsi_c_history])
        # self.handle[8].update(self.time_history, [self.mi_q_history, self.refthetadot_c_history])
        self.handle[7].update(self.time_history, [self.mi_d_a_history])
        self.handle[8].update(self.time_history, [self.mi_d_e_history])
        self.handle[9].update(self.time_history, [self.mi_d_r_history])


class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
           

