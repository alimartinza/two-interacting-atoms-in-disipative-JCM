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

def lanczos_beta2(H, psi0):
    """
    Run two steps of the Lanczos algorithm starting from psi0.
    Returns (beta1, beta2): the two off-diagonal Lanczos coefficients.

    Interpretation:
      beta1 = 0  =>  psi0 is an eigenstate      (Krylov dim = 1)
      beta2 = 0  =>  dynamics confined to 2D subspace (Krylov dim = 2)
      beta2 > 0  =>  full 3D space explored     (Krylov dim = 3)

    Plotting log10(beta2) gives a continuous measure of proximity to
    two-state dynamics: smaller = more confined.
    """
    v  = psi0.astype(complex) / np.linalg.norm(psi0)
    a1 = np.real(v.conj() @ H @ v)
    r  = H @ v - a1 * v
    b1 = np.linalg.norm(r)
    if b1 < 1e-14:
        return b1, 0.0
    v2 = r / b1
    a2 = np.real(v2.conj() @ H @ v2)
    r2 = H @ v2 - a2*v2 - b1*v
    b2 = np.linalg.norm(r2)
    return b1, b2

# ── Parameters ────────────────────────────────────────────────────────────────
n_val, J_val, k_val = 3, 0.0, 0.0
g_val   = 0.01
scale   = 15 * g_val
N_param = 300

D_arr = np.linspace(-scale, scale, N_param)
x_arr = np.linspace(1e-4,   scale, N_param)
D_g   = D_arr / g_val
x_g   = x_arr / g_val
ext   = [D_g[0], D_g[-1], x_g[0], x_g[-1]]

# Compute log10(beta2) for each basis initial state
log_beta2 = np.zeros((N_param, N_param, 3))

for i, x_val in enumerate(x_arr):
    for j, D_val in enumerate(D_arr):
        H = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
        for b in range(3):
            psi0 = np.zeros(3, dtype=complex); psi0[b] = 1.0
            _, b2 = lanczos_beta2(H, psi0)
            log_beta2[i, j, b] = np.log10(b2 + 1e-16)

print(f"log10(beta2) range: [{log_beta2.min():.2f}, {log_beta2.max():.2f}]")

# ── Degeneracy lines ──────────────────────────────────────────────────────────
xg_line = np.linspace(0, x_g[-1], 300)
def clip(Dg, xg, lo, hi):
    m = (Dg >= lo) & (Dg <= hi); return Dg[m], xg[m]
def add_lines(ax, legend=True):
    for slope, ls, lab in [(3,'--',r'$D=3x$'),(4,'-',r'$D=4x$'),(5,':',r'$D=5x$')]:
        dc, xc = clip(slope*xg_line, xg_line, D_g[0], D_g[-1])
        ax.plot(dc, xc, 'w', ls=ls, lw=1.5, label=lab)
    if legend:
        ax.legend(fontsize=8, loc='upper left')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(r'$\log_{10}(\beta_2)$ — second Lanczos coefficient  ($g=0.01,\ n=3,\ J=k=0$)'
             '\nsmall = near 2D invariant subspace = two-state dynamics', fontsize=12)

labels = [r'$|\psi_0\rangle=|1\rangle$',
          r'$|\psi_0\rangle=|2\rangle$',
          r'$|\psi_0\rangle=|3\rangle$']

for col in range(3):
    ax   = axes[col]
    data = log_beta2[:,:,col]
    cm   = ax.imshow(data, origin='lower', extent=ext, aspect='auto', cmap='viridis')
    cb   = plt.colorbar(cm, ax=ax)
    cb.set_label(r'$\log_{10}(\beta_2)$', fontsize=10)
    # Contour where beta2 < 0.1*g  (practical two-state threshold)
    ax.contour(D_g, x_g, data,
               levels=[np.log10(0.1 * g_val)],
               colors='red', linewidths=2, linestyles='--')
    add_lines(ax, legend=(col==0))
    ax.set_xlabel(r'$D/g$', fontsize=12)
    ax.set_ylabel(r'$x/g$', fontsize=12)
    ax.set_title(labels[col], fontsize=11)

plt.tight_layout()
plt.savefig('graficos/orbits/03_lanczos_beta2_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lanczos_beta2_map.png")
