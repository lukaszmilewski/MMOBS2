import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt

# **********Parametry symulacji**********
no_of_steps = 3000
N = 3
B = 0
J = 1
betas = np.linspace(0.2, 2, 10)


# **********inicjalizacja*************
def inicjacja_siatki():
    s = np.empty(N * N, dtype=np.int).reshape(N, N)
    for i in range(N):
        for j in range(N):
            s[i][j] = random.choice([-1, 1])
    return s


def inicjacja_szachownicy(s):
    blacks = ((s // N + s % N) % 2).astype(bool)
    whites = np.logical_not(blacks)
    return blacks, whites


# ***************** GŁOWNY PROGRAM *************************

def liczmagnetyzacje(s):
    m = np.roll(s, 1, axis=0)
    m += np.roll(s, -1, axis=0)
    m += np.roll(s, 1, axis=1)
    m += np.roll(s, -1, axis=1)
    return m


def ising(s, beta):
    M_lista = []
    blacks, whites = inicjacja_szachownicy(s)
    for t in range(no_of_steps):
        m = liczmagnetyzacje(s)
        # energia
        E_plus = -m
        # prawdopodobienstwo
        p_plus = 1 / (1 + np.exp(2 * beta * E_plus))
        r = np.random.rand(N, N)
        up = r < p_plus
        # wybor ktore aktualizuje
        wybor = random.choice([blacks, whites])
        s[wybor] = -1
        s[wybor & up] = +1

        if t >= 1500 and t % 2 == 0:
            M = (np.abs(np.mean(s)))
            M_lista.append(M)

    M_srednie = np.mean(M_lista)
    chi = np.var(M_lista) * beta
    return M_srednie, chi


def plots(M, chi, m_error, chi_error):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].set_title('Wykres bezwzględnej wartości magnetycznej od wartości temperatury odwrotnej')
    axs[0].set_xlabel('beta')
    axs[0].set_ylabel('magnetyzacja M')
    axs[0].errorbar(betas, M, m_error)
    axs[1].set_title('Wykres podatności magnetycznej od wartości temperatury odwrotnej')
    axs[1].set_xlabel('beta')
    axs[1].set_ylabel('Podatność magnetyczna chi')
    axs[1].errorbar(betas, chi, chi_error)

    fig.savefig('3_Ising_lm_wykresy')
    plt.show()
    plt.close()


def main():
    s = inicjacja_siatki()
    M_koncowe = []
    chi_koncowe = []
    M_koncowe_error = []
    chi_koncowe_error = []

    for beta in betas:
        M_srednie_lista = []
        lista_chi = []
        for k in range(10):
            M_srednie, chi = ising(s, beta)
            M_srednie_lista.append(M_srednie)
            lista_chi.append(chi)

        M_koncowe.append(np.mean(M_srednie_lista))
        chi_koncowe.append(np.mean(lista_chi))
        M_koncowe_error.append(np.std(M_srednie_lista))
        chi_koncowe_error.append(np.std(lista_chi))

    plots(M_koncowe, chi_koncowe, M_koncowe_error, chi_koncowe_error)


if __name__ == '__main__':
    main()
