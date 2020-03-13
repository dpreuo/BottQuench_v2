import create_ham as ch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import linalg as la
import time
import functions as fn
import create_ham as cHam
import datetime
from matplotlib.animation import FuncAnimation
sbn.set()

#the quesion here is - does the degree of localisation of the states influence the values of the bott index? maybe it has something to do with the places where bott index diverges?

Lx = 150
Ly = 70
u = 1.21

disorder_parameter = 0.

ks = np.linspace(0, 2 * np.pi, Ly + 1)[:-1]
xs_arr = np.arange(Lx)
ys_arr = np.arange(Ly)
u_values=np.random.normal(u,disorder_parameter,Lx)

plt.plot(u_values)
plt.show()

ipr_values = fn.calculate_localisation_parameter(u_values,ks,True)

n = np.arange(2*Lx)

n_grid,k_grid = np.meshgrid(n,ks/np.pi)

plt.pcolor(n_grid,k_grid,ipr_values.real)
plt.colorbar()
plt.xlabel('individual states')
plt.ylabel('k value')
plt.show()

# plt.pcolor(ipr_values.imag)
# plt.colorbar()
# plt.show()

# bott_values = bott_test(u_values,ks,False)

# plt.plot(bott_values)
# plt.show()