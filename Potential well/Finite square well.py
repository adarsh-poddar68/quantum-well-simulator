#Finite square well potential
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = -10**-10   # Initial position
rn = 10**-10    # Final position
n = 1000        # Number of steps

x = np.linspace(r0, rn, n-1)
delx = x[1] - x[0]
print(f"x[401] = {x[401]}, x[601] = ", x[601])
print("L = ", x[601] - x[401])

# Constants
hbar = 6.582e-16  # Reduced Planck’s constant in eV·s
m = 9.11e-31      # Electron mass in kg
f = (hbar**2) / (2 * m * delx**2)
eV = 1.6 * 10**-19  # eV in Joules
V0 = 5 * eV         # Potential step height in Joules

V = np.zeros((n-1))

# Setting potential values for specific regions
for i in range(401):
    V[i] = V0

for j in range(601, n-1):
    V[j] = V0

# Matrix construction for Hamiltonian
D = np.zeros((n-1, n-1))
np.fill_diagonal(D, 2 / delx**2)
np.fill_diagonal(D[1:], -1 / delx**2)
np.fill_diagonal(D[:, 1:], -1 / delx**2)

# Hamiltonian matrix
D = D * hbar**2 / (2 * m) + np.diag(V)

print('Hamiltonian matrix:\n', D[4:4, :4])

# Solve eigenvalue problem
E_ev, psi = np.linalg.eigh(D)
E_ev = E_ev * hbar**2 / (2 * m * hbar**2)

# Plot the first few eigenfunctions (Probability Densities)
plt.figure(figsize=(15, 6))
for i in range(3):
    plt.plot(x, psi[:, i]**2, label=f'Energy level {i+1}: E_ev = {E_ev[i]:.3e}')
plt.axhline(y=0, color='black')
plt.xlabel('x')
plt.ylabel(r'$\Psi^2$', size=15)
plt.title('Probability for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()

# Plot the wavefunctions
plt.figure(figsize=(15, 6))
for i in range(3):
    plt.plot(x, psi[:, i], label=f'Energy level {i+1}: E_ev = {E_ev[i]:.3e}')
plt.axhline(y=0, color='black')
plt.xlabel('x')
plt.ylabel(r'$\Psi$', size=15)
plt.title('Wave Functions for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()
