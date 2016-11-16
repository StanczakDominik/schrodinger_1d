import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# params
NG = 100
n = 1
omega = 5 * np.pi**2 / 2
kappa = 10
# NT = 40000
# T, dt = np.linspace(0, 1, NT, retstep=True)

dt = 0.0001
T = np.arange(0, 4, dt)
NT = T.size
x, dx = np.linspace(0, 1, NG, retstep=True)
psi_R = 2**0.5 * np.sin(n * np.pi * x)
psi_I = np.zeros_like(psi_R)
# plane wave
# psi = np.exp(1j*1*x*np.pi*2)
# psi_R, psi_I = psi.real, psi.imag

def hamiltonian(psi, t):
    H = np.zeros(NG, dtype=float)
    H[1:-1] = -0.5 * (psi[2:] + psi[:-2] - 2 * psi[1:-1]) / dx**2 +\
        kappa * (x[1:-1]-0.5) * psi[1:-1] * np.sin(omega*t)
    return(H)

psi_R_history = np.zeros((NT, NG))
psi_I_history = np.zeros((NT, NG))
ampl2_history = np.zeros((NT, NG))
params_history = np.zeros((NT, 3))
for i, t in enumerate(T):
    amplitude2 = (psi_R**2 + psi_I**2)
    norm = dx * amplitude2.sum()
    x_avg = dx * (x*amplitude2).sum()
    energy = dx * (psi_R * hamiltonian(psi_R, t) +\
                   psi_I * hamiltonian(psi_I, t)).sum()

    ampl2_history[i] = amplitude2
    psi_R_history[i] = psi_R
    psi_I_history[i] = psi_I
    params_history[i] = norm, x_avg, energy

    psi_R += 0.5 * dt * hamiltonian(psi_I, t)
    psi_I -= dt * hamiltonian(psi_R, t + 0.5 * dt)
    psi_R += 0.5 * dt * hamiltonian(psi_I, t + dt)


fig, ax = plt.subplots()
ax.set_ylim(min((psi_R_history.min(), psi_I_history.min())), max((psi_R_history.max(), psi_I_history.max(), ampl2_history.max())))
ampline, = ax.plot(x, amplitude2)
psi_R_line, = ax.plot(x, psi_R)
psi_I_line, = ax.plot(x, psi_I)
diag_string = "i: {} t: {:.2f} norm: {:.2f} x_avg: {:.2f} energy: {:.2f}"
text = ax.text(dx, 0.1, diag_string.format(i, t, norm, x_avg, energy))

PLOT_EVERY_X_STEPS = 10
def animate(i):
    i = i * PLOT_EVERY_X_STEPS
    ampline.set_ydata(ampl2_history[i])
    psi_R_line.set_ydata(psi_R_history[i])
    psi_I_line.set_ydata(psi_I_history[i])
    text.set_text(diag_string.format(i, T[i], *params_history[i]))
    return [ampline, psi_R_line, psi_I_line, text]

anim = animation.FuncAnimation(fig, animate, frames=NT // PLOT_EVERY_X_STEPS, interval=10)
# anim.save("animacja.mp4")
plt.show()

fig1, ax1 = plt.subplots()
norm_time, x_time, energy_time = params_history.T
ax1.plot(T, norm_time, label="norm")
ax1.plot(T, x_time, label="x")
ax1.plot(T, energy_time, label="energy")
ax1.legend()
plt.show()
