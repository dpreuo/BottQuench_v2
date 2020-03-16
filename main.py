import create_ham as ch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import functions as fn
import time
import create_ham as cHam
import datetime
from matplotlib.animation import FuncAnimation

Lx = 150
Ly = 70
ui = 1.5
uf = 2.2
tf = 100
Nt = 1000



ks = np.linspace(0, 2 * np.pi, Ly + 1)[:-1]
xs_arr = np.arange(Lx)
ys_arr = np.arange(Ly)
u_values_i = xs_arr * 0 + ui
u_values_f = xs_arr * 0 + uf

t_values = np.linspace(0, tf, Nt)

bott_values = np.zeros((Nt, Lx))

t0 = time.time()
for i in range(Nt):
    bott_values[i, :] = fn.find_bott_index_at_time_old_method(u_values_i, u_values_f, ks, t_values[i], edges_tf=True)
    ti = time.time()
    t_left = (ti - t0) * (Nt / (i + 1) - 1)

    print(datetime.timedelta(seconds=t_left), ' left')

np.save('results/bott_values', bott_values)

fig, ax = plt.subplots(figsize=(5, 3))
ax.set(ylim=(-4, 4))
line = ax.plot(xs_arr, bott_values[0, :], color='k', lw=2)[0]


def animate(i):
    line.set_ydata(bott_values[i, :])


anim = FuncAnimation(
    fig, animate, interval=100, frames=len(t_values) - 1)
plt.draw()

bott_name = 'bott'
anim.save('animations/' + bott_name + '.mp4')
