import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt

# **********Simulation Parameters**********
no_of_steps = 3000
N = 3
B = 0
J = 1
betas = np.linspace(0.2, 2, 10)


# **********Initialization*************
def initialize_grid():
    s = np.empty(N * N, dtype=np.int).reshape(N, N)
    for i in range(N):
        for j in range(N):
            s[i][j] = random.choice([-1, 1])
    return s


def initialize_checkerboard(s):
    blacks = ((s // N + s % N) % 2).astype(bool)
    whites = np.logical_not(blacks)
    return blacks, whites


# ***************** MAIN PROGRAM *************************

def calculate_magnetization(s):
    m = np.roll(s, 1, axis=0)
    m += np.roll(s, -1, axis=0)
    m += np.roll(s, 1, axis=1)
    m += np.roll(s, -1, axis=1)
    return m


def ising(s, beta):
    M_list = []
    blacks, whites = initialize_checkerboard(s)
    for t in range(no_of_steps):
        m = calculate_magnetization(s)
        # energy
        E_plus = -m
        # probability
        p_plus = 1 / (1 + np.exp(2 * beta * E_plus))
        r = np.random.rand(N, N)
        up = r < p_plus
        # choosing which to update
        choice = random.choice([blacks, whites])
        s[choice] = -1
        s[choice & up] = +1

        if t >= 1500 and t % 2 == 0:
            M = (np.abs(np.mean(s)))
            M_list.append(M)

    avg_M = np.mean(M_list)
    chi = np.var(M_list) * beta
    return avg_M, chi


def plots(M, chi, m_error, chi_error):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].set_title('Plot of absolute magnetization against inverse temperature')
    axs[0].set_xlabel('beta')
    axs[0].set_ylabel('Magnetization M')
    axs[0].errorbar(betas, M, m_error)
    axs[1].set_title('Plot of magnetic susceptibility against inverse temperature')
    axs[1].set_xlabel('beta')
    axs[1].set_ylabel('Magnetic susceptibility chi')
    axs[1].errorbar(betas, chi, chi_error)

    fig.savefig('3_Ising_lm_plots')
    plt.show()
    plt.close()


def main():
    s = initialize_grid()
    final_M = []
    final_chi = []
    final_M_error = []
    final_chi_error = []

    for beta in betas:
        avg_M_list = []
        chi_list = []
        for k in range(10):
            avg_M, chi = ising(s, beta)
            avg_M_list.append(avg_M)
            chi_list.append(chi)

        final_M.append(np.mean(avg_M_list))
        final_chi.append(np.mean(chi_list))
        final_M_error.append(np.std(avg_M_list))
        final_chi_error.append(np.std(chi_list))

    plots(final_M, final_chi, final_M_error, final_chi_error)


if __name__ == '__main__':
    main()
