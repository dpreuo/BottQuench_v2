import create_ham as ch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import linalg as la
import time
import functions as fn
import create_ham as cHam
import functions as func
import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sbn.set()

L = 40
u_in  = 1.7
u_out = 2.4
cutoff = 0.5
disorder_parameter = 0.

delta = (2 * np.pi) / L
N_total = L * L * 2

xs_arr = np.arange(L)
ys_arr = np.arange(L)

X_grid, Y_grid = np.meshgrid(xs_arr, ys_arr)

# u_values = np.zeros((L , L )) + u
edg = False
# u_values = np.random.normal(u,disorder_parameter,(L, L ))


u_values =  ((X_grid - L/2).__abs__() <= L/4 )*((Y_grid - L/2).__abs__() <= L/4)*u_in + (((X_grid - L/2).__abs__() > L/4 ) + ((Y_grid - L/2).__abs__() > L/4))*u_out


# plt.pcolor(u_values)
# plt.colorbar()
# plt.show()

xs = np.zeros(N_total)
ys = np.zeros(N_total)
for i in range(N_total):
    _, xs[i], ys[i] = func.number_to_index(i, L)
X_exp = np.diag(np.exp(1j * xs * delta))
Y_exp = np.diag(np.exp(1j * ys * delta))
X_exp_star = np.diag(np.exp(-1j * xs * delta))
Y_exp_star = np.diag(np.exp(-1j * ys * delta))

H = cHam.create_full_hamiltonian(u_values, edges=edg)

# plt.pcolor(H.__abs__())
# plt.show()

P = H * 0

eigenvalues, eigenvectors = la.eigh(H)
for j in range(N_total):
    P += np.outer(eigenvectors[:, j], np.conj(eigenvectors[:, j])) if eigenvalues[j] <= 0 else 0

UVUV = np.linalg.multi_dot([P, X_exp, P, Y_exp, P, X_exp_star, P, Y_exp_star, P])
M = UVUV + np.eye(N_total, N_total) - P

T, Z = la.schur(M)
U, s_vals ,V = la.svd(M)

eigs = np.diag(T)
nubers = np.arange(N_total)

plt.subplot(1,2,1)
thetas = np.linspace(0, 2 * np.pi, 1000)
x_circ = np.cos(thetas)
y_circ = np.sin(thetas)
plt.scatter((eigs).real, (eigs).imag)
plt.plot(x_circ+1, y_circ)
plt.plot(x_circ * cutoff, y_circ * cutoff)

plt.subplot(1,2,2)
plt.plot(eigs.__abs__())
plt.plot(s_vals[::-1])
plt.show()


cancel_matrix = np.ones((N_total,N_total))
for i in range(N_total):
    if (1-eigs[i]).__abs__() >=1 or eigs[i].__abs__() <= cutoff:
        cancel_matrix[i,:] = 0
        cancel_matrix[:,i] = 0
        print(i)



T_fixed = T*cancel_matrix + np.eye(N_total)*(1-cancel_matrix)

plt.pcolor(T_fixed.real>= 1e-8)
plt.show()

M_fixed = np.linalg.multi_dot([Z, T_fixed, Z.conj().T])

next = la.logm(M_fixed)
out = np.diag(next) * L * L
bott_index_tr = np.imag(out[::2] + out[1::2]) / (2 * np.pi)
bott_new_method = bott_index_tr.reshape([L, L])

print('new method done')

# this finds the bott index using the old (dumb) method!
ns = la.null_space(UVUV, rcond = cutoff)
sh = ns.shape
Q = np.zeros((N_total,N_total), dtype= 'complex')
for n in range(sh[1]):
    Q += np.outer(ns[:, n], np.conj(ns[:, n]))
next = la.logm(UVUV + Q*1000000)
out = np.diag(next) * L * L
bott_index_tr = np.imag(out[::2] + out[1::2]) / (2 * np.pi)
bott_old_method = bott_index_tr.reshape([L, L])

x_vector = np.arange(L)
y_vector = np.arange(L)
x_grid, y_grid = np.meshgrid(x_vector, y_vector)

print('old method done')


# # this finds the bott index using the shit method!
# ns = la.null_space(UVUV)
# sh = ns.shape
# Q = np.zeros((N_total,N_total), dtype= 'complex')
# for n in range(sh[1]):
#     Q += np.outer(ns[:, n], np.conj(ns[:, n]))
# next = la.logm(UVUV + Q*1000000)
# out = np.diag(next) * L * L
# bott_index_tr = np.imag(out[::2] + out[1::2]) / (2 * np.pi)
# bott_shit_method = bott_index_tr.reshape([L, L])
#
# print('shit method done')

z_grid = bott_new_method
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.title('New Method')
plt.draw()

# z_grid = bott_shit_method
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
#                 linewidth=0, antialiased=False)
# ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
#                 linewidth=0, antialiased=False)
# plt.title('Shit Method')
# plt.draw()


z_grid = bott_old_method
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.title('Old Method')
plt.show()

