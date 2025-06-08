# Créé par mathe, le 07/06/2025 en Python 3.7
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # Pour diagonalisation

# Constantes physiques (unités naturelles)
hbar = 1.0
m = 1.0
a = 1.0       # largeur du puits
V0 = 50.0     # profondeur du puits (positif, potentiel = -V0 dans le puits)

# Discrétisation de l’espace
L = 5 * a                  # domaine spatial total
N = 1000
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Définition du potentiel
V = np.zeros_like(x)
V[np.abs(x) <= a / 2] = -V0

# Construction du Hamiltonien (méthode des différences finies)
diag = np.full(N, hbar**2 / (m * dx**2)) + V
off_diag = np.full(N - 1, -hbar**2 / (2 * m * dx**2))
H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Diagonalisation
energies, wavefuncs = eigh(H)

# Sélection des états non liés (E > 0)
free_indices = np.where(energies >= 0)[0][:5]  # afficher les 5 premiers états non liés

# Normalisation et tracé
plt.figure(figsize=(12, 6))
plt.plot(x, V, 'k--', label='Potentiel $V(x)$')

for i in free_indices:
    psi = wavefuncs[:, i]
    E = energies[i]
    norm = np.sqrt(np.trapz(np.abs(psi)**2, x))  # normalisation dans le domaine
    psi /= norm
    plt.plot(x, psi + E, '--', label=f"État libre $E = {E:.2f}$")

plt.title("États non liés (états de diffusion) dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("Énergie / Fonction d’onde")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

