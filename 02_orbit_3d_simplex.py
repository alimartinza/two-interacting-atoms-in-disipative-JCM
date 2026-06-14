import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def H_matrix(x_val, g_val, D_val, J_val, k_val, n_val):
    n = n_val
    return np.array([
        [x_val*(n-2)**2 + D_val + J_val,  np.sqrt(2*(n-1))*g_val,  0],
        [np.sqrt(2*(n-1))*g_val,           x_val*(n-1)**2 - J_val + 2*k_val, np.sqrt(2*n)*g_val],
        [0,                                np.sqrt(2*n)*g_val,       x_val*n**2 - D_val + J_val]
    ])

def evolve_pops(H, psi0, t_arr):
    evals, evecs = np.linalg.eigh(H)
    c    = evecs.conj().T @ psi0
    pops = np.zeros((len(t_arr), 3))
    for ti, t in enumerate(t_arr):
        psi_t    = evecs @ (c * np.exp(-1j * evals * t))
        pops[ti] = np.abs(psi_t)**2
    return pops, evals, c

# ── Parameters ────────────────────────────────────────────────────────────────
n_val, J_val, k_val = 3, 0.0, 0.0
g_val = 0.01
t_arr = np.linspace(0, 2000, 80000)

cases = [
    (0.0,    0.0,   1, r'$D=x=0,\ |\psi_0\rangle=|2\rangle$  [2-state]',          'plasma'),
    (0.0,    0.0,   0, r'$D=x=0,\ |\psi_0\rangle=|1\rangle$  [3-state]',          'viridis'),
    (3*0.05, 0.05,  0, r'$D=3x,\ x/g=5,\ |\psi_0\rangle=|1\rangle$  [near 2-state]', 'inferno'),
]

fig = plt.figure(figsize=(18, 6))
fig.suptitle(r'Orbits in probability simplex $(P_1, P_2, P_3)$,  $P_1+P_2+P_3=1$'
             rf'  ($g={g_val},\ n=3$)', fontsize=13)

for idx, (D_val, x_val, psi_idx, label, cmap) in enumerate(cases):
    H    = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
    psi0 = np.zeros(3, dtype=complex); psi0[psi_idx] = 1.0
    pops, evals, c = evolve_pops(H, psi0, t_arr)

    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.scatter(pops[:,0], pops[:,1], pops[:,2],
               c=t_arr, cmap=cmap, s=0.3, alpha=0.4)

    # Draw simplex edges
    corners = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]])
    ax.plot(corners[:,0], corners[:,1], corners[:,2], 'k-', lw=1.5, alpha=0.5)

    # Mark initial state
    ax.scatter(*pops[0], color='red', s=80, zorder=10, label='start')

    ax.set_xlabel(r'$P_1$', fontsize=10, labelpad=1)
    ax.set_ylabel(r'$P_2$', fontsize=10, labelpad=1)
    ax.set_zlabel(r'$P_3$', fontsize=10, labelpad=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_title(label, fontsize=9, pad=4)

    pops_eig = np.abs(c)**2
    Neff = 1/np.sum(pops_eig**2)
    S    = -np.sum(pops_eig * np.log(pops_eig + 1e-15))
    w12  = abs(evals[1]-evals[0])
    w13  = abs(evals[2]-evals[0])
    print(f"{label}")
    print(f"  Neff={Neff:.3f}, S={S:.3f}, evals={np.round(evals,4)}")
    print(f"  omega12={w12:.4f}, omega13={w13:.4f}, ratio={w12/w13:.6f}\n")

plt.tight_layout()
plt.savefig('graficos/orbits/02_orbit_3d_simplex.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: orbit_3d_simplex.png")
