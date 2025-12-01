from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence_ali
import matplotlib as mpl
from entrelazamiento_lib import negativity_hor
import os 

# ES EL HERMANO DE CORREDOR.PY, LA IDEA ES HACER SIMULACIONES CON CORREDOR Y LEERLAS CON LEEDOR.

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
os.chdir(script_path)

e=basis(2,0)
gr=basis(2,1)

N_c=4

e0=tensor(e,basis(N_c,0)) #1
e1=tensor(e,basis(N_c,1)) #2
g0=tensor(gr,basis(N_c,0)) #3
g1=tensor(gr,basis(N_c,1)) #4
g2=tensor(gr,basis(N_c,2)) 

sz=tensor(sigmaz(),qeye(N_c))
sx=tensor(sigmax(),qeye(N_c))
sy=tensor(sigmay(),qeye(N_c))
sp=tensor(sigmap(),qeye(N_c))
sm=tensor(sigmam(),qeye(N_c))
a=tensor(qeye(2),destroy(N_c))

w_0=1
g=0.001*w_0

def omega_n(n_:int,delta:float):
    return np.sqrt(delta**2+4*g**2*n_)

x=0.001*g
gamma=0.1*g
p=0.1*0.1*g
tita=0.0
#drive-download-20251118
#data jcm p0.01 estan mal los nombres-20251119
evals_death_t=np.loadtxt(f'datajcm/evalst death x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita}.txt')
fg_u_delta=np.loadtxt(f'datajcm/fgu x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita}.txt')
fg_d_delta=np.loadtxt(f'datajcm/fgd x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita}.txt')
Nu_delta=np.loadtxt(f'datajcm/Nu x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita}.txt')
Nd_delta=np.loadtxt(f'datajcm/Nd x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita}.txt')
fg_delta=fg_d_delta-fg_u_delta
fg_delta_mean=np.zeros_like(fg_delta)
avg_range=200
for i_delta in range(len(fg_delta)):
    for i_t in range(avg_range,len(fg_delta[0])):    
        fg_delta_mean[i_delta][i_t]=np.mean(fg_delta[i_delta][i_t-avg_range:i_t])

steps=len(fg_u_delta[0])
omega=np.sqrt(4*g**2+(0-x)**2)
'''---Simulacion numerica---'''
T=2*np.pi/omega
t_final=70*T

t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

fig_fg=plt.figure()
ax_fg=fig_fg.add_subplot(projection='3d')
delta_array=np.linspace(-g,g,33)
dELTA, gT = np.meshgrid(delta_array/g, g*t,indexing='ij')
ax_fg.plot_surface(dELTA,gT,fg_delta/np.pi,cmap=mpl.cm.coolwarm)#,rstride=1,cstride=int(steps/10))

# ax_fg.plot_wireframe(dELTA,gT,fg_delta,cstride=len(delta_array))
ax_fg.set_xlabel(r'$\Delta/g$')
ax_fg.set_ylabel(r'$gt$')
ax_fg.set_zlabel(r'$\delta \phi/\pi$')
plt.show()

indice_individual=[0,16,29,30]
fig_individual=plt.figure()
ax_ind=fig_individual.add_subplot()
ax_ind.set_title(fr'$\chi={x/g}g ; \gamma={gamma/g}g ; p={p/g}g$')
ax_ind.hlines(0,t[0]/T,t[-1]/T,'grey','dashed','0')
colors=['blue','red','orange','green']
for i,ind in enumerate(indice_individual):
    delta=np.linspace(-g,g,33)[ind]
    ax_ind.plot(t/T,fg_u_delta[ind],color=colors[i],label=rf'$\Delta={delta/g}g$')#,linestyle='dashed')
    ax_ind.plot(t/T,fg_d_delta[ind],color=colors[i],linestyle='dashdot')
    ax_ind.vlines(evals_death_t[ind]/T,-10,10,colors[i])

ax_ind.set_xlabel('t/T')
ax_ind.set_ylabel(r'$\phi$')
plt.legend()
plt.show()
plt.close()

fig_individual=plt.figure()
ax_ind=fig_individual.add_subplot()
ax_ind.set_title(fr'$\chi={x/g}g ; \gamma={gamma/g}g ; p={p/g}g$')
ax_ind.hlines(0,t[0]/T,t[-1]/T,'grey','dashed','0')
for i,ind in enumerate(indice_individual):
    delta=np.linspace(-g,g,33)[ind]
    ax_ind.plot(t/T,fg_delta[ind],label=rf'$\Delta={delta/g}g$',color=colors[i])
    ax_ind.plot(t/T,fg_delta_mean[ind],color=colors[i],linestyle='dashed')
    ax_ind.vlines(evals_death_t[ind]/T,-10,10,colors[i])

ax_ind.set_xlabel('t/T')
ax_ind.set_ylabel(r'$\delta\phi$')
plt.legend()
plt.show()