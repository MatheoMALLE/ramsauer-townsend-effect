# Créé par mathe, le 07/06/2025 en Python 3.7
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constantes (unités réduites)
hbar = 1.0
m = 1.0

# Paramètres du puits
V0 = 5.0         # profondeur (potentiel = -V0 dans le puits)
a = 1.0          # largeur du puits
x_min, x_max = -10, 10

# Potentiel
def V(x):
    return -V0 if -a/2 <= x <= a/2 else 0

# Système d'équations pour Schrödinger (forme réduite)
def schrodinger(E):
    def system(x, y):
        psi, phi = y
        return [phi, 2 * m / hbar**2 * (V(x) - E) * psi]
    return system

# Calcul du coefficient de transmission par la méthode de tir
def transmission(E):
    sol = solve_ivp(
        schrodinger(E),
        [x_min, x_max],
        [0.0, 1e-5],  # condition initiale faible (non nulle)
        t_eval=np.linspace(x_min, x_max, 1000)
    )
    psi = sol.y[0]
    x = sol.t

    # Estimation de l'amplitude après la barrière
    psi_out = psi[x > a]
    T = np.max(np.abs(psi_out))**2
    return T

# Balayage des énergies
energies = np.linspace(0.01, 10, 200)
T_vals = np.array([transmission(E) for E in energies])

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(energies, T_vals, label="Transmission $T(E)$")
plt.xlabel("Énergie $E$")
plt.ylabel("Coefficient de transmission $T$")
plt.title("Effet Ramsauer–Townsend : transmission à travers un puits")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

