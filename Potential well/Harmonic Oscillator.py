# Schrodinger equation for Harmonic Oscillator

import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = -10         # initial position
rn = 10          # final position
n = 1000         # number of steps

x = np.linspace(r0, rn, n - 1)
delx = x[1] - x[0]

# Constants
hbar = 1.1973
m = 0.511e6
k = (hbar**2) / (2 * m * delx**2)
w = 1

V = np.zeros((n - 1))

for i in range(n - 1):
    V[i] = 0.5 * w**2 * (x[i]**2)

# Matrix Construction
diagonals = np.zeros((n - 1, n - 1))
np.fill_diagonal(diagonals, 2 / delx**2)
np.fill_diagonal(diagonals[1:], -1 / delx**2)
np.fill_diagonal(diagonals[:, 1:], -1 / delx**2)

# Hamiltonian Matrix
D = diagonals + (2 * m * V / hbar**2)

print('Hamiltonian matrix:\n', D[:4, :4])

# Solve eigenvalue problem
E_ev, psi = np.linalg.eigh(D)
E_ev = (E_ev * hbar**2) / (2 * m)

# Plot the first few eigenfunctions
plt.figure(figsize=(10, 6))

for i in range(3):  # Plot first 3 energy levels
    plt.plot(x, psi[:, i], label=f'Energy level {i+1}: E_ev = {E_ev[i]:.5f}')
plt.axhline(y=0, color='black')
plt.xlabel('x')
plt.ylabel(r'$\Psi$', size=15)
plt.title('Wave Functions for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

for i in range(3):  # Plot first 3 energy levels
    plt.plot(x, psi[:, i]**2, label=f'Energy level {i+1}: E_ev = {E_ev[i]:.5f}')
plt.axhline(y=0, color='black')
plt.xlabel('x')
plt.ylabel(r'$\Psi^2$', size=15)
plt.title('Probability density for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()
