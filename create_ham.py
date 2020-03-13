import matplotlib.pyplot as plt
import numpy as np
import functions as func


def create_hamiltonian(u_values, ky, edges = False):
    Lx = np.size(u_values)

    H = np.zeros((2*Lx, 2*Lx))

    A = np.array([[np.cos(ky), -1j * np.sin(ky)], [1j * np.sin(ky), -np.cos(ky)]])
    B = 0.5 * np.array([[1, 1j], [1j, -1]])
    B_dag = 0.5 * np.array([[1, -1j], [-1j, -1]])
    r_shift = np.roll(np.eye(Lx), 1, axis=1)
    l_shift = np.roll(np.eye(Lx), -1, axis=1)

    H = np.kron(r_shift,B)+ np.kron(l_shift,B_dag)+ np.kron(np.eye(Lx),A)+np.kron(np.diag(u_values),[[1,0],[0,-1]])

    if edges == True:
        H[-1, 0] = H[0, -1] = 0
        H[-2, 0] = H[0, -2] = 0
        H[-1, 1] = H[1, -1] = 0
        H[-2, 1] = H[1, -2] = 0

    return H

def create_full_hamiltonian(u_values, edges= False):
    sh = np.shape(u_values)

    if sh[0] != sh[1] :
        raise Exception('wrong input - use an L by L square matrix for u values')

    L = sh[0]
    N_total = L * L * 2
    H = np.zeros((N_total, N_total), dtype=complex)

    if edges == False:
        for mx in range(L):
            for my in range(L):
                x_mat = (1 / 2) * np.array([[1, 1j], [1j, -1]])  # =np.array([[1,0],[0,0]]) #
                y_mat = (1 / 2) * np.array([[1, 1], [-1, -1]])  # =np.array([[1,0],[0,0]]) #

                for a in range(2):
                    for b in range(2):
                        H[func.index_to_number(a, (mx + 1) % L, my, L), func.index_to_number(b, mx, my, L)] += x_mat[
                            a, b]
                        H[func.index_to_number(a, mx, (my + 1) % L, L), func.index_to_number(b, mx, my, L)] += y_mat[
                            a, b]

                        H[func.index_to_number(a, mx, my, L), func.index_to_number(b, (mx + 1) % L, my, L)] += np.conj(
                            x_mat[b, a])
                        H[func.index_to_number(a, mx, my, L), func.index_to_number(b, mx, (my + 1) % L, L)] += np.conj(
                            y_mat[b, a])

                H[func.index_to_number(0, mx, my, L), func.index_to_number(0, mx, my, L)] += u_values[mx,my]
                H[func.index_to_number(1, mx, my, L), func.index_to_number(1, mx, my, L)] += -u_values[mx,my]

        return (H)


    elif edges == True:

        for mx in range(L):
            for my in range(L - 1):
                x_mat = (1 / 2) * np.array([[1, 1j], [1j, -1]])  # =np.array([[1,0],[0,0]]) #
                y_mat = (1 / 2) * np.array([[1, 1], [-1, -1]])  # =np.array([[1,0],[0,0]]) #

                for a in range(2):
                    for b in range(2):
                        H[func.index_to_number(a, mx, (my + 1) % L, L), func.index_to_number(b, mx, my, L)] += y_mat[
                            a, b]
                        H[func.index_to_number(a, mx, my, L), func.index_to_number(b, mx, (my + 1) % L, L)] += np.conj(
                            y_mat[b, a])

        for mx in range(L - 1):
            for my in range(L):
                x_mat = (1 / 2) * np.array([[1, 1j], [1j, -1]])  # =np.array([[1,0],[0,0]]) #
                y_mat = (1 / 2) * np.array([[1, 1], [-1, -1]])  # =np.array([[1,0],[0,0]]) #

                for a in range(2):
                    for b in range(2):
                        H[func.index_to_number(a, (mx + 1) % L, my, L), func.index_to_number(b, mx, my, L)] += x_mat[
                            a, b]
                        H[func.index_to_number(a, mx, my, L), func.index_to_number(b, (mx + 1) % L, my, L)] += np.conj(
                            x_mat[b, a])

        for mx in range(L):
            for my in range(L):
                H[func.index_to_number(0, mx, my, L), func.index_to_number(0, mx, my, L)] += u_values[mx,my]
                H[func.index_to_number(1, mx, my, L), func.index_to_number(1, mx, my, L)] += -u_values[mx,my]

        return (H)

