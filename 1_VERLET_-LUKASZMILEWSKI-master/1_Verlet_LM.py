import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import trange, tqdm


# -------- klasy ----------

class atom:  # Klasa określająca wszystkie elementy/atomy w układzie: w tym przypadku jest to i planeta i słońce
    def __init__(self, x0, v0, m, f, ):  # trajectory = np.empty(0, dtype=float)):
        # położenie
        self.X0 = x0
        self.X = x0
        self.X_plus1 = None
        self.X_minus1 = None
        # prędkość
        self.V0 = v0
        self.V = v0
        self.V_plus_half = None
        self.V_minus_half = None
        # masa`
        self.m = m
        # Siła działająca na dany element ukladu przez
        # inny element ukladu - np sila grawitacji miedzy planeta a sloncem
        self.F = f
        self.f_next = None
        self.f_prev = None
        # self.trajectory = trajectory
        self.xlist = []
        self.ylist = []

        self.eklist = []
        self.eplist = []
        self.eclist = []
        self.tlist = []

        self.ek = 0
        self.ep = 0

        self.g = 0.01

    def count_force(self, atom2):  # Definicja sily grawitacji, działającej na dany atom(self)
        # wywierana prz ez podany Atom2

        r = atom2.X - self.X  # wektor odleglosci
        r_len = math.sqrt(r[0] ** 2 + r[1] ** 2)
        r3 = r_len ** 3
        self.F = (self.g * atom2.m * self.m / r3) * r
        return self.F

    def count_Ek(self):
        v_len = math.sqrt(self.V[0] ** 2 + self.V[1] ** 2)
        return self.m * v_len ** 2 / 2

    def count_Ep(self, atom2):
        r = atom2.X - self.X
        r_len = math.sqrt(r[0] ** 2 + r[1] ** 2)
        return -self.g * self.m * atom2.m / r_len

    def give_velocity_value(self):
        return self.V

    def give_polozenie(self):
        return self.X

    def give_trajectory(self):
        return self.xlist, self.ylist


def subplots(atom1, atom2, method):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if method == 'e':
        fig.suptitle('Euler')
    if method == 'v':
        fig.suptitle('Verlet')
    if method == 'lf':
        fig.suptitle('Leapfrog')

    axs[0, 0].plot(atom1.xlist, atom1.ylist)
    axs[0, 0].scatter(atom2.X0[0], atom2.X0[1])
    axs[0, 0].set_title('Trajektoria')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    axs[0, 1].plot(atom1.tlist, atom1.eklist)
    axs[0, 1].set_title('Energia kinetyczna')
    axs[0, 1].set_xlabel('t')
    axs[0, 1].set_ylabel('Ek')

    axs[1, 0].plot(atom1.tlist, atom1.eplist)
    axs[1, 0].set_title('Energia potencjalna')
    axs[1, 0].set_xlabel('t')
    axs[1, 0].set_ylabel('Ep')

    axs[1, 1].plot(atom1.tlist, atom1.eclist)
    axs[1, 1].set_title('Energia całkowita')
    axs[1, 1].set_xlabel('t')
    axs[1, 1].set_ylabel('Ec')

    fig.savefig("LM_" + method + "_subplots")
    plt.show()
    plt.close()


class simulation:  # Klasa symulujaca cala symulacje

    def __init__(self, atoms, dt, no_of_steps, method, steps_per_frame=10,
                 export_frames=True, export_subplots=True):
        self.atoms = atoms  # lista atomów biorących udzial w symulacji
        self.dt = dt  # krok czasowy
        self.max_step = no_of_steps
        self.method = method
        self.subplots = export_subplots
        # ----- do animacji -----
        self.step = 0
        self.steps_per_frame = steps_per_frame
        # export parameters
        self.frames = export_frames

        self.x = np.zeros([self.max_step + 1, len(self.atoms)], dtype=np.int16)
        self.y = np.zeros([self.max_step + 1, len(self.atoms)], dtype=np.int16)
        if self.frames:
            self.initiate_frame_print()
        self.main()

    def euler(self, atom1):
        atom1.X = atom1.X + atom1.V * self.dt + atom1.F * self.dt ** 2 / (2 * atom1.m)
        atom1.V = atom1.V + atom1.F * self.dt / atom1.m

    def verlet(self, atom1):

        if atom1.X_plus1 is None:
            atom1.X_plus1 = atom1.X + atom1.V * self.dt + (1 / 2) * (atom1.F / atom1.m) * (self.dt ** 2)

        else:
            atom1.X_plus1 = 2 * atom1.X - atom1.X_minus1 + (atom1.F / atom1.m) * self.dt ** 2
            atom1.V = 1 / 2 * (atom1.X_plus1 - atom1.X_minus1) / self.dt

        atom1.X_minus1 = atom1.X
        atom1.X = atom1.X_plus1

    def leap_frog(self, atom1):

        if atom1.V_minus_half is None:
            atom1.V_minus_half = atom1.V0 - atom1.F / (2 * atom1.m) * self.dt
        atom1.V_plus_half = atom1.V_minus_half + atom1.F / atom1.m * self.dt
        atom1.X = atom1.X + atom1.V_plus_half * self.dt
        atom1.V = (atom1.V_minus_half + atom1.V_plus_half) / 2
        atom1.V_minus_half = atom1.V_plus_half

    def steps(self):
        atom1 = self.atoms[0]
        atom2 = self.atoms[1]
        method = self.method

        atom1.F = atom1.count_force(atom2)

        atom1.xlist.append(atom1.X[0])
        atom1.ylist.append(atom1.X[1])

        if method == 'e':
            self.euler(atom1)

        if method == 'v':
            self.verlet(atom1)

        if method == 'lf':
            self.leap_frog(atom1)

        atom1.ek = atom1.count_Ek()
        atom1.eklist.append(atom1.ek)

        atom1.ep = atom1.count_Ep(atom2)
        atom1.eplist.append(atom1.ep)

        atom1.eclist.append(atom1.ek + atom1.ep)

        if not atom1.tlist:

            atom1.tlist.append(self.dt)
        else:
            atom1.tlist.append(self.dt + atom1.tlist[-1])

    def initiate_frame_print(self):
        self.frame = 0

        # tworzy folder na klatki symulacyjne
        try:
            os.mkdir(self.method + '_frames')
        except:
            pass

    def save_frame(self, name):
        plt.tick_params(axis='both', which='both', direction='in',
                        right=True, top=True)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        for atom, i in zip(self.atoms, range(len(self.atoms))):
            plt.scatter(atom.X[0], atom.X[1])
            plt.plot(self.x[:self.step, i], self.y[:self.step, i], linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0., dpi=300)
        # zamykamy na koniec, by nie zawalać pamięci
        plt.close()

    def main(self):
        for i in trange(self.max_step + 1):
            self.step += 1
            self.steps()
            self.x[i] = self.atoms[0].X[0]
            self.y[i] = self.atoms[0].X[1]

            if self.frames and i % self.steps_per_frame == 0:
                self.save_frame(self.method + '_frames/f_{:05d}.png'.format(self.frame))
                self.frame += 1
        if self.subplots:
            subplots(self.atoms[0], self.atoms[1], self.method)


# -------- parametry symulacji ----------

#       [m,   M, no_of_steps, dt, planetX0, planetV0, sunX0, sunV0]
param = [0.1, 500, 8000, 0.001, np.array([0, 1]), np.array([2, 0]), np.array([0, 0]), np.array([0, 0])]

# -------- obiekty ----------

planet_euler = atom(param[4], param[5], param[0], 0)

planet_verlet = atom(param[4], param[5], param[0], 0)

planet_leapfrog = atom(param[4], param[5], param[0], 0)

sun = atom(param[6], param[7], param[1], 0)

# -------- symulacja --------

simul_euler = simulation([planet_euler, sun], param[3], param[2], 'e')
simul_verlet = simulation([planet_verlet, sun], param[3], param[2], 'v')

simul_leapfrog = simulation([planet_leapfrog, sun], param[3], param[2], 'lf')

print("Koniec")
