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

bott_values = np.load('results/bott_values.npy')

ks = np.linspace(0, 2 * np.pi, Ly + 1)[:-1]
xs_arr = np.arange(Lx)
t_values = np.linspace(0, tf, Nt)

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
