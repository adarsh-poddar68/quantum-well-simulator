# Particle in a box
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = -5  # initial position
rn = 5   # final position
n = 100  # Number of steps
delx = (rn - r0) / n
x = np.linspace(r0, rn, n - 1)

# Constants
hbar = 1.0545718e-34
m = 9.10938356e-31
f = (hbar**2) / (2 * m * delx**2)
V = 0  # we choose potential according to the problem

# Matrix construction
diagonals = np.zeros((n - 1, n - 1))
np.fill_diagonal(diagonals, 2 * f + V)
np.fill_diagonal(diagonals[1:], -f)
np.fill_diagonal(diagonals[:, 1:], -f)

# Hamiltonian matrix
print("Hamiltonian matrix: \n", diagonals[:3, :3])

# Solve eigenvalue problem
E, psi = np.linalg.eigh(diagonals)
E_ev = E * 6.242e18

# Plot the first few eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(3):  # Plotting the first 3 energy levels
    plt.plot(x, psi[:, i]**2, label=f'Energy level {i + 1}: E_ev={E_ev[i]:.3e}')
plt.axhline(y=0, color='black')
plt.xlabel('x')
plt.ylabel(r'$\Psi^2$', size=15)
plt.title('Wave functions for different Energy levels')
plt.legend()
plt.grid()
plt.show()

# Calculate exact energies and plot wavefunctions
L = rn - r0  # Length of the box
plt.figure(figsize=(10, 6))
for i in range(1, 4):  # for n = 1, 2, 3
    psi_exact = np.sqrt(2 / L) * np.sin(i * np.pi * (x - r0) / L)  # Exact wavefunction
    plt.plot(x, psi_exact**2, label=f'Exact Energy Function for n={i}')

# Final plotting
plt.xlabel('x (position)')
plt.ylabel(r'$\Psi^2$', size=15)
plt.title('Exact Wavefunctions for Particle in a Box')
plt.legend()
plt.grid()
plt.show()
