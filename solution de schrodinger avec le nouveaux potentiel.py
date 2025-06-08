# Créé par mathe, le 08/06/2025 en Python 3.7
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # Pour diagonalisation

# Constantes physiques (unités réduites)
hbar = 1.0   # constante de Planck réduite
m = 1.0      # masse de la particule
a = 1.0      # largeur du puits
V0 = 50.0    # profondeur du puits (positif, mais le potentiel est -V0)

# Discrétisation de l'espace
L = 5 * a                  # domaine total (plus grand que le puits)
N = 1000                   # nombre de points
x = np.linspace(-L, L, N)  # grille spatiale
dx = x[1] - x[0]           # pas d'espace

# Définition du potentiel V(x)
V = -V0 * np.exp(-x**2 / a**2)

# Construction du Hamiltonien (méthode des différences finies)
diag = np.full(N, hbar**2 / (m * dx**2)) + V
off_diag = np.full(N - 1, -hbar**2 / (2 * m * dx**2))
H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Diagonalisation du Hamiltonien
energies, wavefuncs = eigh(H)

# Séparation des états liés et non liés
bound_indices = np.where(energies < 0)[0]
free_indices = np.where(energies >= 0)[0][:3]  # quelques états libres

# Normalisation des fonctions d'onde
for i in range(wavefuncs.shape[1]):
    norm = np.sqrt(np.trapz(np.abs(wavefuncs[:, i])**2, x))
    wavefuncs[:, i] /= norm

# Tracé du potentiel et des fonctions d'onde
plt.figure(figsize=(12, 6))
plt.plot(x, V, 'k--', label='Potentiel $V(x)$')

# États liés (E < 0)
for i in bound_indices:
    psi = wavefuncs[:, i]
    E = energies[i]
    plt.plot(x, psi + E, label=f"État lié $n={i}$, E = {E:.2f}")

# États de diffusion (E > 0)
for i in free_indices:
    psi = wavefuncs[:, i]
    E = energies[i]
    plt.plot(x, psi + E, '--', label=f"État libre $E = {E:.2f}$")

plt.title("États liés et états de diffusion dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("Énergie / Fonction d’onde")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

