import matplotlib.pyplot as plt
import numpy as np
import create_ham as cHam
from scipy import linalg as la
import time
import datetime
import scipy.linalg as la

def index_to_number(a, x, y, L):
    return a + 2 * x + 2 * L * y

def number_to_index(i, L):
    return (i % 2, (i // 2) % L, i // (2 * L) % L)


def find_bott_index_old_method(u_values, k_values, edges = False):
    Lx = u_values.size
    Ly = k_values.size

    edg = edges

    xs_arr = np.arange(Lx)

    eiX = np.diag(np.exp((np.kron(xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))
    eiX_minus = np.diag(np.exp((np.kron(- xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))


    # create the hamiltonian on the other side of the array
    H_last = cHam.create_hamiltonian(u_values, k_values[-1], edges = edg)
    P_last = H_last * 0

    if True:#la.det(H_last) == 0: # if the hamiltoian has a null space you have to use the slow method to produce P
        eigs_last, vecs_last = la.eigh(H_last)
        for j in range(Lx*2):
            P_last += np.outer(vecs_last[:, j], np.conj(vecs_last[:, j])) if eigs_last[j] <= 0 else 0
    else:   #otherwise you use the fast method
        P_last = (1 / 2) * (np.eye(2 * Lx) - np.dot(la.inv(H_last), la.sqrtm(np.dot(H_last, H_last))))

    matrix_out = np.zeros((Lx * 2, Lx * 2), dtype='complex')


    for i in range(Ly):
        print(i+1, ' of ', Ly)
        #create the current hamiltonian
        H_current = cHam.create_hamiltonian(u_values, k_values[i],edges = edg)
        P_current = H_current * 0

        if True:#la.det(H_current) == 0:  # if the hamiltonian has a null space you have to use the slow method to produce P
            eigs_current, vecs_current = la.eigh(H_current)
            for j in range(Lx * 2):
                P_current += np.outer(vecs_current[:, j], np.conj(vecs_current[:, j])) if eigs_current[j] <= 0 else 0
        #this next method is kinda janky --  sometimes it works but it fucks shit up sometimes :(
        else:  # otherwise you use the fast method
            P_current = (1 / 2) * (np.eye(2 * Lx) - np.dot(la.inv(H_current), la.sqrtm(np.dot(H_current, H_current))))

        UVUV = np.linalg.multi_dot([P_current, P_last, eiX, P_last, P_current, eiX_minus, P_current])
        ns = la.null_space(UVUV)
        sh = ns.shape
        Q = 0 * H_current
        for n in range(sh[1]):
            Q += np.outer(ns[:, n], np.conj(ns[:, n]))

        matrix_out += -la.logm(UVUV + Q)
        P_last = P_current

    bout = np.diag(matrix_out)
    bott_index = -Lx * np.imag(bout[::2] + bout[1::2]) / (2 * np.pi)

    return(bott_index)

def find_bott_index_new_method(u_values, k_values, edges = False, cutoff = 0.5):
    Lx = u_values.size
    Ly = k_values.size

    edg = edges

    xs_arr = np.arange(Lx)

    eiX = np.diag(np.exp((np.kron(xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))
    eiX_minus = np.diag(np.exp((np.kron(- xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))


    # create the hamiltonian on the other side of the array
    H_last = cHam.create_hamiltonian(u_values, k_values[-1], edges = edg)
    P_last = H_last * 0

    eigs_last, vecs_last = la.eigh(H_last)
    for j in range(Lx*2):
        P_last += np.outer(vecs_last[:, j], np.conj(vecs_last[:, j])) if eigs_last[j] <= 0 else 0

    matrix_out = np.zeros((Lx * 2, Lx * 2), dtype='complex')

    for i in range(Ly):
        print(i+1, ' of ', Ly)
        H_current = cHam.create_hamiltonian(u_values, k_values[i],edges = edg)
        P_current = H_current * 0

        eigs_current, vecs_current = la.eigh(H_current)
        for j in range(Lx * 2):
            P_current += np.outer(vecs_current[:, j], np.conj(vecs_current[:, j])) if eigs_current[j] <= 0 else 0

        UVUV = np.linalg.multi_dot([P_current, P_last, eiX, P_last, P_current, eiX_minus, P_current])

        M = UVUV + (np.eye(2*Lx)) - P_current

        T, Q = la.schur(M)



        cancel_matrix = np.ones((2*Lx,2*Lx))
        for j in range(2*Lx):
            if (1 - T[i,i]).__abs__() >= 1 or T[i,i].__abs__() <= cutoff:
                cancel_matrix[i, :] = 0
                cancel_matrix[:, i] = 0

        T_fixed = T * cancel_matrix + np.eye(2*Lx) * (1 - cancel_matrix)

        D = T_fixed * np.eye(2 * Lx)
        N = T_fixed - D

        plt.pcolor(D)

        matrix_out += -np.linalg.multi_dot([Q, np.diag(np.log(np.diag(D))) ,Q.conj().T]) + np.dot(N , np.diag(1/(np.diag(D))))

        # matrix_out += -la.logm(UVUV + Q)
        # P_last = P_current

    bout = np.diag(matrix_out)
    bott_index = -Lx * np.imag(bout[::2] + bout[1::2]) / (2 * np.pi)

    return(bott_index)

def find_bott_index_at_time_old_method(u_values_initial, u_values_final, k_values, t, edges_tf = False):
    Lx = u_values_initial.size
    Ly = k_values.size

    edg = edges_tf

    xs_arr = np.arange(Lx)

    eiX = np.diag(np.exp((np.kron(xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))
    eiX_minus = np.diag(np.exp((np.kron(- xs_arr, [1, 1])) * 1j * 2 * np.pi / Lx))

    # create the hamiltonian on the other side of the array
    H_last_initial = cHam.create_hamiltonian(u_values_initial, k_values[-1], edges=edg)
    H_last_final = cHam.create_hamiltonian(u_values_final, k_values[-1], edges=edg)

    eiH_last = la.expm(1j*H_last_final*t)
    eiH_minus_last = la.expm(-1j*H_last_final*t)


    P_last = H_last_initial * 0

    eigs_last, vecs_last = la.eigh(H_last_initial)
    for j in range(Lx * 2):
        P_last += np.outer(vecs_last[:, j], np.conj(vecs_last[:, j])) if eigs_last[j] <= 0 else 0

    P_last_at_time = np.linalg.multi_dot([eiH_last,P_last,eiH_minus_last])

    P_current = H_last_final*0
    matrix_out = np.zeros((Lx * 2, Lx * 2), dtype='complex')

    for i in range(Ly):
        H_current_initial = cHam.create_hamiltonian(u_values_initial, k_values[i], edges=edg)
        H_current_final = cHam.create_hamiltonian(u_values_final, k_values[i], edges=edg)

        eiH_current = la.expm(1j * H_current_final * t)
        eiH_minus_current = la.expm(-1j * H_current_final * t)

        eigs_current, vecs_current = la.eigh(H_current_initial)
        P_current = H_last_final * 0
        for j in range(Lx * 2):
            P_current += np.outer(vecs_current[:, j], np.conj(vecs_current[:, j])) if eigs_current[j] <= 0 else 0

        P_current_at_time = np.linalg.multi_dot([eiH_current, P_current, eiH_minus_current])

        # plt.pcolor((P_current_at_time-P_current).__abs__())
        # plt.colorbar()
        # plt.show()

        UVUV_time = np.linalg.multi_dot([P_current_at_time, eiX, P_current_at_time, P_last_at_time, eiX_minus,P_last_at_time, P_current_at_time])
        ns = la.null_space(UVUV_time,1e-8)
        sh = ns.shape
        Q = 0 * H_current_final
        for n in range(sh[1]):
            Q += np.outer(ns[:, n], np.conj(ns[:, n]))

        matrix_out += la.logm(UVUV_time + Q)

        P_last = P_current
        P_last_at_time = P_current_at_time



    bout = np.diag(matrix_out)
    bott_index = -Lx * np.imag(bout[::2] + bout[1::2]) / (2 * np.pi)

    return (bott_index)

def calculate_localisation_parameter(u_values, k_values, edges = False):
    Lx = u_values.size
    Ly = k_values.size

    edg = edges
    xs_arr = np.arange(Lx)

    X = np.diag((np.kron(xs_arr, [1, 1])) )
    xs_arr = np.arange(Lx)
    ipr = np.zeros((Ly,2*Lx),dtype= 'complex')
    energy_all_values = np.zeros((Ly,Lx*2))

    for i in range(Ly):
        H_current = cHam.create_hamiltonian(u_values, k_values[i], edges=edg)
        energies, eigs = la.eigh(H_current)

        energy_all_values[i , :] = energies

        for j in range(2*Lx):
            ipr[i,j] = np.sum( np.abs(eigs[:,j]**4))

        # plt.plot(ipr[i,:])
        # plt.show()

    # plt.plot(energy_all_values)
    # plt.show()

    return ipr