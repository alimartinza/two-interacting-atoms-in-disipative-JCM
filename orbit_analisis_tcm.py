import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# QUTRIT ORBIT ANALYSIS
# Hamiltonian:
#   H = [[x(n-2)^2 + D + J,  sqrt(2(n-1))*g,  0              ],
#        [sqrt(2(n-1))*g,     x(n-1)^2 - J+2k, sqrt(2n)*g     ],
#        [0,                  sqrt(2n)*g,        x*n^2 - D + J ]]
#
# Diagnostics:
#   - Eigenvalues vs D (for several g values)
#   - Discriminant Delta = prod_{i<j} (eps_i - eps_j)^2
#   - N_eff = 1 / sum_k |c_k|^4   (effective number of participating eigenstates)
#   - S(rho_bar) = -sum_k |c_k|^2 log|c_k|^2  (diagonal ensemble entropy)
#   - Population dynamics P_i(t)
#   - Orbit projection in (P1, P3) plane
# ─────────────────────────────────────────────────────────────────────────────


# ── Hamiltonian ───────────────────────────────────────────────────────────────
def H_matrix(x_val, g_val, D_val, J_val, k_val, n_val):
    n = n_val
    return np.array([
        [x_val*(n-2)**2 + D_val + J_val,  np.sqrt(2*(n-1))*g_val,  0],
        [np.sqrt(2*(n-1))*g_val,           x_val*(n-1)**2 - J_val + 2*k_val,  np.sqrt(2*n)*g_val],
        [0,                                 np.sqrt(2*n)*g_val,      x_val*n**2 - D_val + J_val]
    ])


# ── Orbit diagnostics ────────────────────────────────────────────────────── ----
def get_diagnostics(x_val, g_val, D_val, J_val, k_val, n_val, psi0):
    """
    Returns eigenvalues, N_eff, diagonal-ensemble entropy S,
    minimum eigenvalue gap, and the two independent frequencies.
    """
    H = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
    evals, evecs = np.linalg.eigh(H)
    
    #Agregar Lanczos para tambien hacer un heatmap comparando con N_eff y ver si son parecidos. Aca solo mirariamos el numero de coeficientes de lanczos que son no nulos.


    c = evecs.conj().T @ psi0          # overlaps c_k = <E_k|psi0>
    populations = np.abs(c)**2         # |c_k|^2 — conserved by time evolution

    N_eff  = 1.0 / np.sum(populations**2) #Dimension efectiva de la dinamica
    S      = -np.sum(populations * np.log(populations + 1e-15)) #Esto supuestamente tambien nos sirve para ver la dinemsion de la dinamica -> S=0 es 1 estado, S=ln2 son 2 estados activos, y S=ln3 son 3 estados igualmente activos.
    gaps   = [abs(evals[1]-evals[0]), abs(evals[2]-evals[1]), abs(evals[2]-evals[0])] #frecuencias del sistema
    discriminant_val=gaps[0]**2*gaps[1]**2*gaps[2]**2
    delta_min = min(gaps) 
    omega12   = abs(evals[1] - evals[0]) 
    omega13   = abs(evals[2] - evals[0])

    return evals, N_eff, S, delta_min, omega12, omega13


def discriminant(x_val, g_val, D_val, J_val, k_val, n_val):
    """
    Discriminant Delta = prod_{i<j} (eps_i - eps_j)^2.
    Delta = 0  <=>  two eigenvalues coincide (true degeneracy).
    """
    evals = np.linalg.eigvalsh(H_matrix(x_val, g_val, D_val, J_val, k_val, n_val))
    return ((evals[1]-evals[0])**2 *
            (evals[2]-evals[0])**2 *
            (evals[2]-evals[1])**2)


def evolve_pops(x_val, g_val, D_val, J_val, k_val, n_val, psi0, cycle_number,time_points):
    """
    Returns population matrix pops[t, state] = |<state|psi(t)>|^2,
    plus eigenvalues and overlap coefficients.
    """
    H      = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
    evals, evecs = np.linalg.eigh(H) #devuelve evecs en columnas, es decir evecs[:,k] es el k-esimo autovector
    gaps   = [abs(evals[1]-evals[0]), abs(evals[2]-evals[1]), abs(evals[2]-evals[0])]
    w_max=max(gaps)
    T=2*np.pi/w_max
    t_arr=np.linspace(0,cycle_number*T,time_points)

    c      = evecs.conj().T @ psi0 #esto parece raro, pero el comportamiento de @ es el correcto, transforma el vector (que es fila) en columna y luego hace la multiplicacion correcta.
    pops   = np.zeros((len(t_arr), 3))
    for ti, t in enumerate(t_arr):
        psi_t      = evecs @ (c * np.exp(-1j * evals * t))
        pops[ti]   = np.abs(psi_t)**2
    return pops, evals, c ,t_arr


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                 FIXED PARAMETERS                                            │
# └─────────────────────────────────────────────────────────────────────────────┘
n_val  = 2
J_val  = 0.0
k_val  = 0.0
x_val  = 0.0

psi0     = np.array([0., 1., 0.], dtype=complex)
psi0_index=np.argmax(psi0)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Eigenvalues vs D for several values of g                  -----
# ═════════════════════════════════════════════════════════════════════════════
D_scan      = np.linspace(-5, 15, 2000)
g_vals_plot = [0.0, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=False)
fig.suptitle(rf'Eigenvalues vs $D$  ($n={n_val},\ J={J_val},\ x={x_val}$)', fontsize=13)

for ax, g_v in zip(axes, g_vals_plot):
    ev_list = [np.linalg.eigvalsh(H_matrix(x_val, g_v, D_v, J_val, k_val, n_val))
               for D_v in D_scan]
    ev_arr  = np.array(ev_list)   # shape (N, 3)

    for i, col in enumerate(['royalblue', 'tomato', 'seagreen']):
        ax.plot(D_scan, ev_arr[:, i], color=col, lw=1.5, label=rf'$\varepsilon_{i+1}$')

    ax.set_xlabel(r'$D$', fontsize=11)
    ax.set_ylabel(r'$\varepsilon$', fontsize=11)
    ax.set_title(rf'$g = {g_v}$', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/orbits/11_eigenvalues_vs_D.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eigenvalues_vs_D.png")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — (D, g) Discriminant, N_eff, and entropy maps in (D, g) space ----
# ═════════════════════════════════════════════════════════════════════════════
N       = 400
D_arr   = np.linspace(-5, 15, N)
g_arr   = np.linspace(0.01, 4, N) # ultrastrong coupling ?

disc_map = np.zeros((N, N))
Neff_map = np.zeros((N, N))
S_map    = np.zeros((N, N))

for i, D_val in enumerate(D_arr):
    for j, g_val in enumerate(g_arr):
        disc_map[j, i] = discriminant(x_val, g_val, D_val, J_val, k_val, n_val)
        H              = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
        evals, evecs   = np.linalg.eigh(H)
        # psi0           = np.array([1., 0., 0.], dtype=complex)
        c              = evecs.conj().T @ psi0
        pops           = np.abs(c)**2
        Neff_map[j, i] = 1.0 / np.sum(pops**2)
        S_map[j, i]    = -np.sum(pops * np.log(pops + 1e-15))

ext      = [D_arr[0], D_arr[-1], g_arr[0], g_arr[-1]]
log_disc = np.log10(disc_map + 1e-6)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(rf'$n={n_val},\ J={J_val},\ x={x_val},\ |\psi_0\rangle=|{psi0_index}\rangle$', fontsize=13)

## Discriminant ----
cm1 = axes[0].imshow(log_disc, origin='lower', extent=ext, aspect='auto', cmap='hot_r')
axes[0].set_title(r'$\log_{10}(\Delta)$ — discriminant' + '\n(dark = near degeneracy $\Delta=0$)', fontsize=10)
axes[0].set_xlabel(r'$D$'); axes[0].set_ylabel(r'$g$')
axes[0].contour(D_arr, g_arr, log_disc, levels=[-2], colors='cyan', linewidths=1.5, linestyles='--')
plt.colorbar(cm1, ax=axes[0])

## N_eff ----
cm2 = axes[1].imshow(Neff_map, origin='lower', extent=ext, aspect='auto',
                      cmap='plasma', vmin=1, vmax=3)
axes[1].set_title(r'$N_{\rm eff} = 1/\sum_k|c_k|^4$', fontsize=11)
axes[1].set_xlabel(r'$D$'); axes[1].set_ylabel(r'$g$')
axes[1].contour(D_arr, g_arr, Neff_map, levels=[1.95, 2.05], colors='white', linewidths=1.2)
axes[1].contour(D_arr, g_arr, log_disc, levels=[-2], colors='cyan', linewidths=1.2, linestyles='--')
plt.colorbar(cm2, ax=axes[1])

## Entropy ----
cm3 = axes[2].imshow(S_map, origin='lower', extent=ext, aspect='auto',
                      cmap='viridis', vmin=0, vmax=np.log(3))
axes[2].set_title(r'$S(\bar\rho) = -\sum_k |c_k|^2 \ln|c_k|^2$', fontsize=10)
axes[2].set_xlabel(r'$D$'); axes[2].set_ylabel(r'$g$')
cb3 = plt.colorbar(cm3, ax=axes[2])
cb3.set_ticks([0, np.log(2), np.log(3)])
cb3.set_ticklabels([r'$0$', r'$\ln 2$', r'$\ln 3$'])
axes[2].contour(D_arr, g_arr, log_disc, levels=[-2], colors='cyan', linewidths=1.2, linestyles='--')

plt.tight_layout()
plt.savefig('graficos/orbits/12_dg_discriminant_and_Neff_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: (D,g) discriminant_and_Neff_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — (D, x) Discriminant, N_eff, and entropy maps in (D, x) space ----
# ═════════════════════════════════════════════════════════════════════════════

n_val=2
g_val=0.01

N       = 1000
D_arr   = np.linspace(-5*g_val, 15*g_val, N)
x_arr   = np.linspace(0, 4*g_val, N)

disc_map = np.zeros((N, N))
Neff_map = np.zeros((N, N))
S_map    = np.zeros((N, N))


for i, D_val in enumerate(D_arr):
    for j, x_val in enumerate(x_arr):
        disc_map[j, i] = discriminant(x_val, g_val, D_val, J_val, k_val, n_val)
        H              = H_matrix(x_val, g_val, D_val, J_val, k_val, n_val)
        evals, evecs   = np.linalg.eigh(H)
        c              = evecs.conj().T @ psi0
        pops           = np.abs(c)**2
        Neff_map[j, i] = 1.0 / np.sum(pops**2)
        S_map[j, i]    = -np.sum(pops * np.log(pops + 1e-15))

ext      = [D_arr[0]/g_val, D_arr[-1]/g_val, x_arr[0]/g_val, x_arr[-1]/g_val]
log_disc = np.log10(disc_map + 1e-6)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(rf'$n={n_val},\ J={J_val}, g={g_val},\ |\psi_0\rangle=|{psi0_index}\rangle$', fontsize=13)

# Discriminant
cm1 = axes[0].imshow(log_disc, origin='lower', extent=ext, aspect='auto', cmap='hot_r')
axes[0].set_title(r'$\log_{10}(\Delta)$ — discriminant' + '\n(dark = near degeneracy $\Delta=0$)', fontsize=10)
axes[0].set_xlabel(r'$D/g$'); axes[0].set_ylabel(r'$\chi/g$')
axes[0].contour(D_arr/g_val, x_arr/g_val, log_disc, levels=[-2], colors='cyan', linewidths=1.5, linestyles='--')
plt.colorbar(cm1, ax=axes[0])
axes[0].plot(D_arr/g_val,(D_arr/g_val+2*(k_val/g_val-J_val/g_val))/((2*n_val-1)),color='black',alpha=0.5)
axes[0].plot(D_arr/g_val,(D_arr/g_val)/((2*n_val-2)),color='black',alpha=0.5)
axes[0].plot(D_arr/g_val,(D_arr/g_val-2*(k_val/g_val-J_val/g_val))/((2*n_val-3)),color='black',alpha=0.5)
axes[0].set_ylim(x_arr[0]/g_val,x_arr[-1]/g_val)

# N_eff
cm2 = axes[1].imshow(Neff_map, origin='lower', extent=ext, aspect='auto',
                      cmap='plasma', vmin=1, vmax=3)
axes[1].set_title(r'$N_{\rm eff} = 1/\sum_k|c_k|^4$', fontsize=11)
axes[1].set_xlabel(r'$D/g$'); axes[1].set_ylabel(r'$\chi/g$')
axes[1].contour(D_arr/g_val, x_arr/g_val, Neff_map, levels=[1], colors='white', linewidths=1.2, linestyles='--')
axes[1].contour(D_arr/g_val, x_arr/g_val, Neff_map, levels=[2], colors='black', linewidths=1.2, linestyles='--')
axes[1].contour(D_arr/g_val, x_arr/g_val, Neff_map, levels=[3], colors='yellow', linewidths=1.2, linestyles='--')
axes[1].plot(D_arr/g_val,(D_arr/g_val+2*(k_val/g_val-J_val/g_val))/((2*n_val-1)),color='black',alpha=0.5)
axes[1].plot(D_arr/g_val,(D_arr/g_val)/((2*n_val-2)),color='black',alpha=0.5)
axes[1].plot(D_arr/g_val,(D_arr/g_val-2*(k_val/g_val-J_val/g_val))/((2*n_val-3)),color='black',alpha=0.5)
# axes[1].contour(D_arr, g_arr, log_disc, levels=[-2], colors='cyan', linewidths=1.2, linestyles='--')
plt.colorbar(cm2, ax=axes[1])
axes[1].set_ylim(x_arr[0]/g_val,x_arr[-1]/g_val)


# Entropy
cm3 = axes[2].imshow(S_map, origin='lower', extent=ext, aspect='auto',
                      cmap='viridis', vmin=0, vmax=np.log(3))
axes[2].set_title(r'$S(\bar\rho) = -\sum_k |c_k|^2 \ln|c_k|^2$', fontsize=10)
axes[2].set_xlabel(r'$D/g$'); axes[2].set_ylabel(r'$\chi/g$')
cb3 = plt.colorbar(cm3, ax=axes[2])
cb3.set_ticks([0, np.log(2), np.log(3)])
cb3.set_ticklabels([r'$0$', r'$\ln 2$', r'$\ln 3$'])
axes[2].contour(D_arr/g_val, x_arr/g_val, log_disc, levels=[-2], colors='cyan', linewidths=1.2, linestyles='--')
axes[2].contour(D_arr/g_val, x_arr/g_val, S_map, levels=[np.log(2)], colors='white', linewidths=1.2)
axes[2].contour(D_arr/g_val, x_arr/g_val, S_map, levels=[np.log(3)], colors='white', linewidths=1.2)

plt.tight_layout()
plt.savefig('graficos/orbits/13_dx_discriminant_and_Neff_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: (D,x) discriminant_and_Neff_map.png")

# # ═════════════════════════════════════════════════════════════════════════════
# # PLOT 4 — (D, g) N_eff, entropy, and minimum gap in (D, g) space  [original panel] ----
# # ═════════════════════════════════════════════════════════════════════════════
N2       = 300
D_arr2   = np.linspace(-10, 10, N2)
g_arr2   = np.linspace(0.01, 5, N2)

Neff_map2 = np.zeros((N2, N2))
S_map2    = np.zeros((N2, N2))
gap_map2  = np.zeros((N2, N2))

for i, D_val in enumerate(D_arr2):
    for j, g_val in enumerate(g_arr2):
        evals, N_eff, S, delta_min, _, _ = get_diagnostics(
            x_val, g_val, D_val, J_val, k_val, n_val, psi0)
        Neff_map2[j, i] = N_eff
        S_map2[j, i]    = S
        gap_map2[j, i]  = delta_min

ext2 = [D_arr2[0], D_arr2[-1], g_arr2[0], g_arr2[-1]]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(rf'Orbit diagnostics: $n={n_val},\ J={J_val},\ k={k_val},\ x={x_val},\ |\psi_0\rangle=|{psi0_index}\rangle$',
             fontsize=13, y=1.02)

cm1 = axes[0].imshow(Neff_map2, origin='lower', extent=ext2, aspect='auto',
                      cmap='plasma', vmin=1, vmax=3)
axes[0].set_xlabel(r'$D$', fontsize=12); axes[0].set_ylabel(r'$g$', fontsize=12)
axes[0].set_title(r'$N_{\rm eff} = 1/\sum_k |c_k|^4$', fontsize=12)
plt.colorbar(cm1, ax=axes[0])
axes[0].contour(D_arr2, g_arr2, Neff_map2, levels=[1.95, 2.05], colors='white', linewidths=1)


cm2 = axes[1].imshow(S_map2, origin='lower', extent=ext2, aspect='auto',
                      cmap='viridis', vmin=0, vmax=np.log(3))
axes[1].set_xlabel(r'$D$', fontsize=12); axes[1].set_ylabel(r'$g$', fontsize=12)
axes[1].set_title(r'$S(\bar\rho) = -\sum_k |c_k|^2 \ln|c_k|^2$', fontsize=12)
cb2 = plt.colorbar(cm2, ax=axes[1])
cb2.set_ticks([0, np.log(2), np.log(3)])
cb2.set_ticklabels([r'$0$', r'$\ln 2$', r'$\ln 3$'])

cm3 = axes[2].imshow(np.log10(gap_map2 + 1e-10), origin='lower', extent=ext2, aspect='auto',
                      cmap='hot_r', vmin=-3, vmax=1)
axes[2].set_xlabel(r'$D$', fontsize=12); axes[2].set_ylabel(r'$g$', fontsize=12)
axes[2].set_title(r'$\log_{10}(\min_{i\neq j}|\varepsilon_i - \varepsilon_j|)$' +
                  '\n(dark = degeneracy)', fontsize=12)
plt.colorbar(cm3, ax=axes[2])

plt.tight_layout()
plt.savefig('graficos/orbits/14_orbit_diagnostics_panel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: orbit_diagnostics_panel.png")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Population dynamics and orbit projections for three regimes ----
# ═════════════════════════════════════════════════════════════════════════════

n_val=2
g_val=0.01

colors = ['royalblue', 'tomato', 'seagreen']
slabels = [r'$|1\rangle$', r'$|2\rangle$', r'$|3\rangle$']

cases = [
    (0.8*g_val, 5*g_val),
    (0.5*g_val, 2.5*g_val),
    (1.25*g_val, g_val*2.5),
]

fig= plt.figure(figsize=(13, 10))
fig.suptitle(rf'Population dynamics and orbit projections ($n={n_val},\ J={J_val},\ g={g_val}$)', fontsize=13)
ax1 = fig.add_subplot(3,2,1)
for row, (x_v, D_v) in enumerate(cases):
    pops, evals, c,t_arr = evolve_pops(x_v, g_val, D_v, J_val, k_val, n_val, psi0, 80,10000)
    pops_basis     = np.abs(c)**2
    Neff = 1 / np.sum(pops_basis**2)
    S    = -np.sum(pops_basis * np.log(pops_basis + 1e-15))
    disc = ((evals[1]-evals[0])**2 * (evals[2]-evals[0])**2 * (evals[2]-evals[1])**2)

    title=fr'$x={x_v},\ D={D_v}$: ($N_{{\rm eff}} \approx {Neff:.3f}$)'

    # Left panel: populations vs time
    
    ax_t = fig.add_subplot(3,2,2*row+1) 
    for i in range(3):
        ax_t.plot(t_arr, pops[:, i], color=colors[i], lw=1.0, label=slabels[i])
    ax_t.set_ylabel('Population', fontsize=10)
    ax_t.set_xlabel(r'$t$', fontsize=10)
    ax_t.set_title(title, fontsize=10)
    ax_t.legend(fontsize=9, loc='upper right')
    ax_t.set_ylim(-0.02, 1.05)
    info = (f'$N_{{\\rm eff}}={Neff:.3f}$,  $S={S:.3f}$\n'
            f'$\\varepsilon=({evals[0]:.3f},\\ {evals[1]:.3f},\\ {evals[2]:.3f})$\n'
            f'$\\Delta={disc:.2e}$')
    ax_t.text(0.02, 0.97, info, transform=ax_t.transAxes, fontsize=8,
              va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    # Right panel: orbit projection onto (P1, P3)
    ax_o = fig.add_subplot(3,2,2*row+2,projection='3d')
    sc = ax_o.scatter(pops[:, 0], pops[:,1],pops[:, 2], c=t_arr,
                       cmap='plasma', s=0.5, alpha=0.6)
    # sc01 = ax_o.scatter(pops[:, 0], pops[:, 1], c=t_arr,
    #                    cmap='viridis', s=0.5, alpha=0.6)
    ax_o.set_xlabel(r'$P_1 = |\langle 1|\psi(t)\rangle|^2$', fontsize=10)
    ax_o.set_ylabel(r'$P_2 = |\langle 2|\psi(t)\rangle|^2$', fontsize=10)
    ax_o.set_zlabel(r'$P_3 = |\langle 3|\psi(t)\rangle|^2$', fontsize=10)
    ax_o.set_title('Orbit in $(P_1, P_3)$ plane', fontsize=10)
    ax_o.set_xlim(-0.02, 1.02)
    ax_o.set_ylim(-0.02, 1.02)
    plt.colorbar(sc, ax=ax_o, label=r'$t$')

plt.tight_layout()
plt.savefig('graficos/orbits/15_dynamics_and_orbits.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dynamics_and_orbits.png")


