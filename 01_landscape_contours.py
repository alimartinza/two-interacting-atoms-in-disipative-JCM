import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def H_matrix(x_val, g_val, D_val, J_val, k_val, n_val):
    n = n_val
    return np.array([
        [x_val*(n-2)**2 + D_val + J_val,  np.sqrt(2*(n-1))*g_val,  0],
        [np.sqrt(2*(n-1))*g_val,           x_val*(n-1)**2 - J_val + 2*k_val, np.sqrt(2*n)*g_val],
        [0,                                np.sqrt(2*n)*g_val,       x_val*n**2 - D_val + J_val]
    ])

def psi0_from_angles(t1, t2, phi1, phi2):
    return np.array([np.cos(t1),
                     np.sin(t1)*np.cos(t2)*np.exp(1j*phi1),
                     np.sin(t1)*np.sin(t2)*np.exp(1j*phi2)], dtype=complex)

def compute_Neff_batch(evecs, psi0_batch):
    C    = evecs.conj().T @ psi0_batch
    pops = np.abs(C)**2
    return 1.0 / np.sum(pops**2, axis=0)

# ── Parameters ────────────────────────────────────────────────────────────────
n_val, J_val, k_val = 3, 0.0, 0.0
g_val = 0.01

N_ang  = 80
N_phi  = 30
t1_arr   = np.linspace(0, np.pi/2, N_ang)
t2_arr   = np.linspace(0, np.pi/2, N_ang)
phi1_arr = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
phi2_arr = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
T1, T2   = np.meshgrid(t1_arr, t2_arr, indexing='ij')

interesting_points = [
    (0.0,    0.0,    r'$D=0,\ x=0$'),
    (3*0.05, 0.05,   r'$D=3x,\ x/g=5$'),
    (4*0.05, 0.05,   r'$D=4x,\ x/g=5$'),
    (5*0.05, 0.05,   r'$D=5x,\ x/g=5$'),
    (0.0,    0.10,   r'$D=0,\ x/g=10$'),
]

fig, axes = plt.subplots(1, len(interesting_points), figsize=(20, 5))
fig.suptitle(r'$N_{\rm eff}(\theta_1,\theta_2)$ averaged over $(\phi_1,\phi_2)$  '
             rf'($g={g_val},\ n=3$)', fontsize=13)

for ax, (D_val, x_val, label) in zip(axes, interesting_points):
    H            = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
    evals, evecs = np.linalg.eigh(H)

    Neff_accum = np.zeros((N_ang, N_ang))
    for phi1 in phi1_arr:
        for phi2 in phi2_arr:
            psi_batch = np.array([
                np.cos(T1).ravel(),
                np.sin(T1).ravel() * np.cos(T2).ravel() * np.exp(1j*phi1),
                np.sin(T1).ravel() * np.sin(T2).ravel() * np.exp(1j*phi2)
            ], dtype=complex)
            Neff_accum += compute_Neff_batch(evecs, psi_batch).reshape(N_ang, N_ang)
    Neff_mean = Neff_accum / (N_phi * N_phi)

    ext_s = [t2_arr[0], t2_arr[-1], t1_arr[0], t1_arr[-1]]
    cm = ax.imshow(Neff_mean, origin='lower', extent=ext_s,
                   aspect='auto', cmap='plasma', vmin=1, vmax=3)
    cs = ax.contour(t2_arr, t1_arr, Neff_mean,
                    levels=[1.2, 2.0, 2.8],
                    colors=['white', 'cyan', 'yellow'], linewidths=2.0)
    ax.clabel(cs, fmt={1.2: 'N≈1', 2.0: 'N=2', 2.8: 'N≈3'}, fontsize=9)
    plt.colorbar(cm, ax=ax).set_label(r'$N_{\rm eff}$', fontsize=9)
    ax.set_xlabel(r'$\theta_2$', fontsize=11)
    ax.set_ylabel(r'$\theta_1$', fontsize=11)
    ax.set_xticks([0, np.pi/4, np.pi/2])
    ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'], fontsize=8)
    ax.set_yticks([0, np.pi/4, np.pi/2])
    ax.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'], fontsize=8)
    ax.set_title(label, fontsize=10)
    for t2b, t1b, blab in [(0, 0, r'$|1\rangle$'),
                            (0, np.pi/2, r'$|2\rangle$'),
                            (np.pi/2, np.pi/2, r'$|3\rangle$')]:
        ax.plot(t2b, t1b, 'w*', ms=12, zorder=5)
        ax.text(t2b+0.04, t1b+0.04, blab, color='white', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos/orbits/01_landscape_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: landscape_contours.png")
