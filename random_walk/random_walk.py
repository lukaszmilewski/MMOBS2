import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os
from tqdm import trange
import subprocess as sub
import sys

tlist_to_plot_f = []


class atom(object):  # Class defining all elements/atoms in the system: in this case it's both a planet and a sun
    def __init__(self, v0, m, X, name=None):  # trajectory = np.empty(0, dtype=float)):
        self.name = name
        # position
        self.X = X #np.array([0.5,0.5])
        # for randomizing position without repetitions

        self.Xx = 0
        self.Xy = 0

        self.V0 = v0
        self.V = v0
        self.V_plus_half = None
        self.V_minus_half = None
        self.v_len = None

        self.m = m
        self.F = np.array([0, 0])
        self.kinetic_energy = []
        self.potential_energy = []
        self.full_energy = []
        self.xlist = []
        self.xarray = np.array([])
        self.yarray = np.array([])
        self.ylist = []
        self.eclist = []

        self.ek = None
        self.ep = None

        self.debug_ek_list = []
        self.debug_v_list = []

        self.listaf = []


def initiate_frame_print():
    # creates a folder for simulation frames
    try:
        os.mkdir('frames')
    except:
        pass


class simulation(object):  # Class simulating the entire simulation
    def __init__(self, dt=0.0001, num_of_atoms=16, no_of_steps=4000,
                 steps_per_frame=100, export_frames=True, export_subplots=True,
                 boxsize=8.0, T0=2.5, thermostat=False, temp_term=5, export_debug_plots=True):

        #self.atoms = np.array([atom(np.random.uniform(low=-1, high=1, size=2), 1, np.array([2,1])),
        #                       atom(np.random.uniform(low=-1, high=1, size=2), 1, np.array([3, 1])),
        #                       atom(np.random.uniform(low=-1, high=1, size=2), 1, np.array([3, 3])),
        #                       atom(np.random.uniform(low=-1, high=1, size=2), 1, np.array([2, 2]))
        #                       ])
        self.boxsize = boxsize
        self.atoms = np.array([atom(np.random.randint(low=-10, high=10, size=2), 1, np.random.uniform(low=0.1, high=self.boxsize - 0.1, size=2)
)\
                               for n in range(num_of_atoms)])
        self.frame = 0


        self.T0 = T0
        self.thermostat = thermostat
        self.debugplots = export_debug_plots
        self.temp_term = temp_term
        self.num_of_atoms = num_of_atoms
        self.dt = dt
        self.max_step = no_of_steps
        self.subplots = export_subplots
        self.no_of_steps = no_of_steps
        self.eta = None
        # self.temp = temp
        # periodic boundary conditions

        self.debug_forces_list = []
        self.ep = 0

        self.list_used_Xx = []
        self.list_used_Xy = []
        # position matrix
        self.pos_matrix = []

        # distance vectors matrix r
        self.r_vectors_matrix = np.zeros(shape=(self.num_of_atoms,
                                                self.num_of_atoms), dtype=np.object)

        # r_len matrix
        self.r_len_matrix = np.zeros(shape=(self.num_of_atoms,
                                            self.num_of_atoms), dtype=np.float)

        self.forces_matrix = np.zeros(shape=(self.num_of_atoms,
                                             self.num_of_atoms), dtype=np.object)

        self.forces_len_matrix = np.zeros(shape=(self.num_of_atoms,
                                                 self.num_of_atoms), dtype=np.float)

        self.potential_matrix = np.zeros(shape=(self.num_of_atoms,
                                                self.num_of_atoms), dtype=np.float)

        self.tlist = []
        self.eklist = []
        self.eplist = []
        self.eclist = []
        self.eksum = 0
        self.epsum = 0
        self.listF = []
        # ----- for animation -----
        self.step = 0
        self.steps_per_frame = steps_per_frame
        # export parameters
        self.frames = export_frames

        self.x = np.zeros([self.max_step + 1, len(self.atoms)], dtype=np.int16)
        self.y = np.zeros([self.max_step + 1, len(self.atoms)], dtype=np.int16)
        if self.frames:
            initiate_frame_print()

        self.mFlist = []
        self.x_n = np.zeros((2, self.num_of_atoms))
        # self.F = np.zeros((2, self.n_atoms))

    def give_names(self):

        for i, j in enumerate(self.atoms):
            j.name = "atom" + str(i).zfill(2)

    def sys_commands(self):
        try:
            os.mkdir('results')
        except:
            pass
        #sub.run("for i in *.png; do mv $i wyniki/; done")

        if self.frames:
            sub.run("ffmpeg -framerate 20 -i frames/f_%05d.png -c:v libx264 anim.mp4 -y", shell=True)

            #sub.run("for i in *.mp4; do mv $i wyniki/; done")

        exit()

    def periodic(self, print_message=False):
        # periodic boundary conditions
        for n in self.atoms:

            if n.X[0] > self.boxsize or n.X[0] < 0 or n.X[1] > self.boxsize or n.X[1] < 0:
                n.X = n.X % self.boxsize
            else:
                pass
                if print_message:
                    print("for", n, "X[0] > 8.0", "X[0] change to", n.X[0])

            # if n.X[1] > self.boxsize:
            #    n.X[1] = n.X[1] % self.boxsize
            #    if print_message:
            #        print("for", n, "X[1] > 8.0")

    def give_pos(self, print_message=False):
        self.count_r_vectors_matrix(False)
        print("before randomization", self.r_len_matrix)
        for i, n in enumerate(self.atoms):
            for j, m in enumerate(self.atoms):
                if i != j:
                    #if self.r_len_matrix[i][j] <= 1:
                    #    print("initial", self.r_len_matrix[i][j])
                    while self.r_len_matrix[i][j] <= 1:
                        #print("randomizing")
                        n.X = np.random.uniform(low=0.1, high=self.boxsize - 0.1, size=2)
                        self.count_r_vectors_matrix(False)
        self.count_r_vectors_matrix(False)
        print("after randomization", self.r_len_matrix)

    def update_pos_matrix(self):
        # List of positions (nested)
        self.pos_matrix.clear()
        for n in self.atoms:
            self.pos_matrix.append(n.X)
        # print(self.pos_matrix)

    def find_closest(self, i, j, n, m):
        if self.r_vectors_matrix[i][j][0] > self.boxsize / 2:
            self.r_vectors_matrix[i][j] = n.X - np.array([m.X[0] - self.boxsize, m.X[1]])
        elif self.r_vectors_matrix[i][j][0] < -(self.boxsize / 2):
            self.r_vectors_matrix[i][j] = n.X - np.array([m.X[0] + self.boxsize, m.X[1]])
        elif self.r_vectors_matrix[i][j][1] > self.boxsize / 2:
            self.r_vectors_matrix[i][j] = n.X - np.array([m.X[0], m.X[1] - self.boxsize])
        elif self.r_vectors_matrix[i][j][1] < -(self.boxsize / 2):
            self.r_vectors_matrix[i][j] = n.X - np.array([m.X[0], m.X[1] + self.boxsize])
        else:
            pass

    def count_r_vectors_matrix(self, find_closest):
        for i, n in enumerate(self.atoms):
            for j, m in enumerate(self.atoms):
                self.r_vectors_matrix[i][j] = n.X - m.X
                if find_closest:
                    self.find_closest(i, j, n, m)
                self.r_len_matrix[i][j] = math.sqrt(
                    self.r_vectors_matrix[i][j][0] ** 2 + self.r_vectors_matrix[i][j][1] ** 2)

    def initialise_forces(self):
        for m in self.atoms:
            m.F = np.array([0, 0])

    def count_forces(self, rcutoff=2.5, epsilon=1, sigma=1):
        self.update_pos_matrix()
        self.count_r_vectors_matrix(True)

        for i, n in enumerate(self.atoms):
            for j, m in enumerate(self.atoms):
                n.F = np.array([0, 0])

                r = self.r_vectors_matrix[i][j]
                r_len = self.r_len_matrix[i][j]
                if r_len == 0 or r_len > rcutoff:
                    self.forces_matrix[i][j] = np.array([0, 0])
                    self.potential_matrix[i][j] = 0.0
                else:
                    self.forces_matrix[i][j] = ((24 * (epsilon / sigma) * (2 * (sigma / r_len) ** 13 -
                                                                           (sigma / r_len) ** 7)) - (
                                                        24 * (epsilon / sigma) * (2 * (sigma / rcutoff ** 13 -
                                                                                       (sigma / rcutoff) ** 7)))) * \
                                               (r / r_len)

                    self.potential_matrix[i][j] = (4 * (epsilon / sigma) * (2 * (sigma / r_len) ** 12 -
                                                                            (sigma / r_len) ** 6)) - \
                    (4 * (epsilon / sigma) * (2 * (sigma / rcutoff) ** 12 -
                                              (sigma / rcutoff) ** 6))

            n.F = np.sum(self.forces_matrix[i, :])
            self.mFlist.append(n.F)

    def count_energy(self):
        for i, n in enumerate(self.atoms):
            n.v_len = math.sqrt(n.V[0] ** 2 + n.V[1] ** 2)
            n.ek = (n.m * n.v_len ** 2) / 2
            n.ep = np.sum(self.potential_matrix[i, :])

    def trajectory(self):
        for n in self.atoms:
            n.xlist.append(n.X[0])
            n.ylist.append(n.X[1])

    def plots(self, savefig):

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Leapfrog')

        axs[0, 0].set_xlim([0, self.boxsize])
        axs[0, 0].set_ylim([0, self.boxsize])
        for n in self.atoms:
            axs[0, 0].scatter(n.xlist, n.ylist)
        axs[0, 0].set_title('Trajectory')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')

        axs[0, 1].plot(self.tlist, self.eklist)
        axs[0, 1].set_title('Kinetic Energy')
        axs[0, 1].set_xlabel('t')
        axs[0, 1].set_ylabel('Ek')

        axs[1, 0].plot(self.tlist, self.eplist)
        axs[1, 0].set_title('Potential Energy')
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('Ep')

        axs[1, 1].plot(self.tlist, self.eclist)
        axs[1, 1].set_title('Total Energy')
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel('Ec')
        if savefig:
            fig.savefig("_temp_" + "no_thermo" + "_subplots.png")
        # fig.savefig("_temp_" + str(temp) + "_subplots.png")

        plt.show()
        if savefig:
            plt.close()

    def save_frame(self, name):
        plt.tick_params(axis='both', which='both', direction='in',
                        right=True, top=True)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(0, self.boxsize)
        plt.ylim(0, self.boxsize)
        # for n in self.atoms:
        for n, i in zip(self.atoms, range(len(self.atoms))):
            plt.scatter(n.X[0], n.X[1])  # , c='black', s=1500, alpha=0.6)
            # plt.plot(self.x[:self.step, i], self.y[:self.step, i], linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0., dpi=300)
        # closing to save memory
        plt.close()

    def initialise_temperature(self, T0):
        vlist = []
        b = 0
        for n in self.atoms:
            vlist.append(n.V)
            a = sum(vlist)
            v_cm = a / self.num_of_atoms
            n.V = n.V - v_cm
            n.v_len = math.sqrt(n.V[0] ** 2 + n.V[1] ** 2)
            b = b + n.v_len ** 2
            ek_mean = b / self.num_of_atoms
            scaling_factor = math.sqrt(T0 / ek_mean)
            n.V = n.V * scaling_factor

    def thermostat(self, T_term, num_of_dimensions=2, k=1):
        c = 0
        for n in self.atoms:
            vlen = math.sqrt((n.V[0] ** 2 + n.V[1] ** 2))
            c = c + (n.m * vlen ** 2)
        T = (1 / (num_of_dimensions * self.num_of_atoms * k)) * c

        # print(T)
        self.eta = math.sqrt(T_term / T)
        # print(self.eta)
        return T

    def leap_frog(self, thermostat, print_message=False):
        for n in self.atoms:
            if n.V_minus_half is None:
                n.V_minus_half = n.V0 - n.F / (2 * n.m) * self.dt
            if thermostat:
                self.thermostat(self.temp_term)
                n.V_plus_half = ((2 * self.eta) - 1) * n.V_minus_half + (self.eta * n.F / n.m) * self.dt
            if not thermostat:
                n.V_plus_half = n.V_minus_half + n.F / n.m * self.dt

            n.X = n.X + n.V_plus_half * self.dt
            n.V = (n.V_minus_half + n.V_plus_half) / 2
            n.V_minus_half = n.V_plus_half

            if print_message:
                print(n.name, "X", n.X)

    def animation(self, i):
        if self.frames and i % self.steps_per_frame == 0:
            self.save_frame('frames/f_{:05d}.png'.format(self.frame))
            self.frame += 1

        for n in self.atoms:
            n.xlist.append(n.X[0])
            n.ylist.append(n.X[1])

    def data_for_plots(self, i):
        self.eksum = 0
        self.epsum = 0
        for c, n in enumerate(self.atoms):
            self.eksum += n.ek
            self.epsum += n.ep / 2

            self.x[i] = n.X[0]
            self.y[i] = n.X[1]

        self.eklist.append(self.eksum)
        self.eplist.append(self.epsum)
        self.eclist.append(self.epsum + self.eksum)

        if not self.tlist:
            self.tlist.append(self.dt)
        else:
            self.tlist.append(self.dt + self.tlist[-1])

    def debug_plots1(self, i):
        if self.debugplots:
            if i % 500 == 0:
                self.plots(False)

    def steps(self):

        for i in trange(self.no_of_steps):
            self.leap_frog(self.thermostat)
            self.periodic()
            self.count_forces()
            self.count_energy()
            self.data_for_plots(i)
            self.animation(i)
            self.debug_plots1(i)

    def main(self):
        try:
            sub.run("rm -r frames")
        except:
            pass
        self.give_names()
        self.give_pos()
        initiate_frame_print()
        self.initialise_forces()
        self.count_forces()
        self.count_energy()
        self.trajectory()
        self.plots(False)
        self.steps()
        self.plots(True)
        self.sys_commands()


if __name__ == '__main__':
    simulation().main()
