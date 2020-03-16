import matplotlib.pyplot as plt
import numpy as np
import create_ham as cHam
from scipy import linalg as la
import time
import datetime
import scipy.linalg as la
import functions as fn

Lx = 80
Ly = 50
u = 1.5

tf = 100
Nt = 1000

ks = np.linspace(0, 2 * np.pi, Ly + 1)[:-1]
xs_arr = np.arange(Lx)
ys_arr = np.arange(Ly)
u_values = xs_arr * 0 + u


bott_values = fn.find_bott_index_new_method(u_values,ks,False,0.5)

print(bott_values)

plt.plot(bott_values.real)
plt.show()