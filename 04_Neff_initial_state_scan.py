import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def H_matrix(g_val, D_val, n_val):
    """Hamiltonian with x=k=J=0."""
    n = n_val
    return np.array([
        [D_val,                   np.sqrt(2*(n-1))*g_val,  0],
        [np.sqrt(2*(n-1))*g_val,  0,                       np.sqrt(2*n)*g_val],
        [0,                       np.sqrt(2*n)*g_val,      -D_val]
    ])

def make_states(t1_arr, t2_arr, phi1_arr, phi2_arr):
    """
    Build array of all states from the 4-parameter grid.
    Returns psi_batch of shape (3, N_total) and the four index arrays.
    """
    T1, T2, P1, P2 = np.meshgrid(t1_arr, t2_arr, phi1_arr, phi2_arr, indexing='ij')
    psi = np.array([
        np.cos(T1).ravel(),
        np.sin(T1)*np.cos(T2)*np.exp(1j*P1),
        np.sin(T1)*np.sin(T2)*np.exp(1j*P2)
    ], dtype=complex).reshape(3, -1)   # (3, N_total)
    # normalize (should already be unit norm, but numerical safety)
    psi /= np.linalg.norm(psi, axis=0)
    return psi, T1.ravel(), T2.ravel(), P1.ravel(), P2.ravel()

def compute_Neff(evecs, psi_batch):
    C    = evecs.conj().T @ psi_batch   # (3, N)
    pops = np.abs(C)**2
    return 1.0 / np.sum(pops**2, axis=0)

# ── Parameters ────────────────────────────────────────────────────────────────
n_val = 3
g_val = 0.01
scale = 15 * g_val

# Grid over initial state angles
# Fine enough to resolve the level sets
N_t1  = 40
N_t2  = 40
N_phi = 20   # for each of phi1, phi2
t1_arr   = np.linspace(0,        np.pi/2, N_t1)
t2_arr   = np.linspace(0,        np.pi/2, N_t2)
phi1_arr = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
phi2_arr = np.linspace(0, 2*np.pi, N_phi, endpoint=False)

# Build state grid once (same for all D values)
psi_batch, T1f, T2f, P1f, P2f = make_states(t1_arr, t2_arr, phi1_arr, phi2_arr)
N_total = psi_batch.shape[1]
print(f"Total initial states per D value: {N_total}")

# Tolerance for "integer" Neff
# Neff=1: Neff in [1, 1+tol]
# Neff=2: Neff in [2-tol, 2+tol]
# Neff=3: Neff in [3-tol, 3]
tol = 0.15

# D values to scan
D_values = np.linspace(-scale, scale, 25)

# Storage: for each D, store the angle tuples in each level set
results = []   # list of dicts, one per D

for D_val in D_values:
    H            = H_matrix(g_val, D_val, n_val)
    evals, evecs = np.linalg.eigh(H)
    Neff         = compute_Neff(evecs, psi_batch)

    mask1 = (Neff >= 1.0) & (Neff < 1 + tol)
    mask2 = (Neff >= 2 - tol) & (Neff <= 2 + tol)
    mask3 = (Neff >= 3 - tol) & (Neff <= 3.0)

    def extract(mask):
        return np.column_stack([T1f[mask], T2f[mask], P1f[mask], P2f[mask]])

    results.append({
        'D':      D_val,
        'D_over_g': D_val / g_val,
        'evals':  evals,
        'N1':     extract(mask1),
        'N2':     extract(mask2),
        'N3':     extract(mask3),
        'counts': (mask1.sum(), mask2.sum(), mask3.sum()),
    })

print("\nD/g   |  #(N=1)  #(N=2)  #(N=3)")
print("-"*40)
for r in results:
    print(f"{r['D_over_g']:>6.1f}  |  {r['counts'][0]:>6}  {r['counts'][1]:>6}  {r['counts'][2]:>6}")

# ── Figure 1: size of each level set vs D ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
D_g_arr = [r['D_over_g'] for r in results]
c1 = [r['counts'][0] / N_total for r in results]
c2 = [r['counts'][1] / N_total for r in results]
c3 = [r['counts'][2] / N_total for r in results]

ax.plot(D_g_arr, c1, 'o-', color='#4477AA', lw=2, label=r'$N_{\rm eff}\approx 1$')
ax.plot(D_g_arr, c2, 's-', color='#228833', lw=2, label=r'$N_{\rm eff}\approx 2$')
ax.plot(D_g_arr, c3, '^-', color='#EE6677', lw=2, label=r'$N_{\rm eff}\approx 3$')
ax.set_xlabel(r'$D/g$', fontsize=12)
ax.set_ylabel('Fraction of initial states', fontsize=12)
ax.set_title(r'Size of level sets vs $D$  ($x=k=J=0,\ g=0.01,\ n=3$)'
             f'\ntolerance = {tol}', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('levelset_fractions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: levelset_fractions.png")

# ── Figure 2: project level sets onto (theta1, theta2) for several D values ──
# Show how the N=2 manifold moves in angle space as D varies
D_plot_indices = [0, 6, 12, 18, 24]   # 5 representative D values

fig, axes = plt.subplots(3, len(D_plot_indices), figsize=(20, 12))
fig.suptitle(r'Level sets projected onto $(\theta_1, \theta_2)$  '
             r'(each point = one $(\phi_1,\phi_2)$ pair satisfying condition)'
             f'\n$x=k=J=0,\ g={g_val},\ n=3$', fontsize=12)

colors = {1: '#4477AA', 2: '#228833', 3: '#EE6677'}
labels = {1: r'$N_{\rm eff}\approx 1$', 2: r'$N_{\rm eff}\approx 2$', 3: r'$N_{\rm eff}\approx 3$'}
keys   = {1: 'N1', 2: 'N2', 3: 'N3'}

for col, idx in enumerate(D_plot_indices):
    r = results[idx]
    for row, m in enumerate([1, 2, 3]):
        ax  = axes[row, col]
        pts = r[keys[m]]
        if len(pts) > 0:
            ax.scatter(pts[:,1], pts[:,0], s=2, alpha=0.4,
                       color=colors[m], rasterized=True)
        ax.set_xlim(0, np.pi/2); ax.set_ylim(0, np.pi/2)
        ax.set_xticks([0, np.pi/4, np.pi/2])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'], fontsize=8)
        ax.set_yticks([0, np.pi/4, np.pi/2])
        ax.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'], fontsize=8)
        ax.set_xlabel(r'$\theta_2$', fontsize=9)
        ax.set_ylabel(r'$\theta_1$', fontsize=9)
        if row == 0:
            ax.set_title(fr'$D/g={r["D_over_g"]:.1f}$' + '\n' + labels[m], fontsize=9)
        else:
            ax.set_title(labels[m], fontsize=9)
        ax.text(0.05, 0.92, f'n={len(pts)}', transform=ax.transAxes,
                fontsize=8, color=colors[m])
        # Mark basis states
        for t2b, t1b, blab in [(0,0,r'$|1\rangle$'),
                                (0,np.pi/2,r'$|2\rangle$'),
                                (np.pi/2,np.pi/2,r'$|3\rangle$')]:
            ax.plot(t2b, t1b, 'k*', ms=8, zorder=5)

plt.tight_layout()
plt.savefig('levelset_projections.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: levelset_projections.png")

# ── Figure 3: phi1-phi2 structure of the N=2 level set at D=0 ────────────────
# For a fixed (theta1, theta2) near the N=2 region, show how phi1, phi2 vary
r0   = results[12]   # D=0
pts2 = r0['N2']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(r'$N_{\rm eff}\approx 2$ level set at $D=0$: structure in angle space',
             fontsize=12)

axes[0].scatter(pts2[:,1], pts2[:,0], s=3, alpha=0.4, color='#228833')
axes[0].set_xlabel(r'$\theta_2$', fontsize=11); axes[0].set_ylabel(r'$\theta_1$', fontsize=11)
axes[0].set_title(r'Projection onto $(\theta_1,\theta_2)$', fontsize=10)
axes[0].set_xticks([0,np.pi/4,np.pi/2]); axes[0].set_xticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$'])
axes[0].set_yticks([0,np.pi/4,np.pi/2]); axes[0].set_yticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$'])

axes[1].scatter(pts2[:,2], pts2[:,3], s=3, alpha=0.4, color='#228833')
axes[1].set_xlabel(r'$\phi_1$', fontsize=11); axes[1].set_ylabel(r'$\phi_2$', fontsize=11)
axes[1].set_title(r'Projection onto $(\phi_1,\phi_2)$', fontsize=10)

sc = axes[2].scatter(pts2[:,1], pts2[:,2], c=pts2[:,0], cmap='plasma', s=3, alpha=0.5)
plt.colorbar(sc, ax=axes[2]).set_label(r'$\theta_1$', fontsize=10)
axes[2].set_xlabel(r'$\theta_2$', fontsize=11); axes[2].set_ylabel(r'$\phi_1$', fontsize=11)
axes[2].set_title(r'$(\theta_2, \phi_1)$ colored by $\theta_1$', fontsize=10)

plt.tight_layout()
plt.savefig('levelset_N2_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: levelset_N2_structure.png")

# ── Save the arrays ───────────────────────────────────────────────────────────
np.save('levelset_results.npy', np.array(results, dtype=object), allow_pickle=True)
print("Saved: levelset_results.npy  (load with np.load(..., allow_pickle=True))")