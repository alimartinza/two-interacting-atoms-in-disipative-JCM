from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence
from matplotlib import colormaps

e=basis(2,0)
gr=basis(2,1)

e0=tensor(e,basis(2,0)) #1
e1=tensor(e,basis(2,1)) #2
g0=tensor(gr,basis(2,0)) #3
g1=tensor(gr,basis(2,1)) #4

sz=tensor(sigmaz(),qeye(2))
sx=tensor(sigmax(),qeye(2))
sy=tensor(sigmay(),qeye(2))
sp=tensor(sigmap(),qeye(2))
sm=tensor(sigmam(),qeye(2))
a=tensor(qeye(2),destroy(2))


def omega_n(n_:int,delta:float):
    return np.sqrt(delta**2+4*g**2*n_)

fig=plt.figure(figsize=(16,9))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
fig.subplots_adjust(wspace=0.05)

ax=[ax1,ax2]
ax1.set_xlabel('$t/T$',size=20)
ax2.set_xlabel('$t/T$',size=20)
ax1.set_ylabel('Poblaciones',size=20)
ax2.set_yticklabels([])

# esfera1=Bloch()
# esfera1.make_sphere()
# esfera1.clear()

# esfera2=Bloch()
# esfera2.make_sphere()
# esfera2.clear()

# esferas=[esfera1,esfera2]

fig_fg=plt.figure(figsize=(16,9))
ax1_fg=fig_fg.add_subplot(221)
ax2_fg=fig_fg.add_subplot(222)
ax1_concu=fig_fg.add_subplot(223)
ax2_concu=fig_fg.add_subplot(224)
ax_fg=[ax1_fg,ax2_fg]
ax_concu=[ax1_concu,ax2_concu]
colors=colormaps['plasma'](np.linspace(0,1,4))

w_0=1
g=0.001*w_0

gamma=0#0.1*g
p=0#.005*g
delta=0
x=g

for j,gamma in enumerate([0,0.1*g]):
    H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)


    '''---Simulacion numerica---'''
    T=2*np.pi/omega_n(1,delta)
    t_final=25*T
    print(t_final)
    steps=20000
    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp] #operadores de colapso/lindblad
    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    psi0=e0

    sol_d=mesolve(H,psi0,t,c_ops=l_ops)
    concu=concurrence(sol_d.states)
    # prob_g0=[expect(sol_d.states[i],g0*g0.dag()) for i in range(len(sol_d.states))]
    # prob_g1=[expect(sol_d.states[i],g1*g1.dag()) for i in range(len(sol_d.states))]
    # prob_e0=[expect(sol_d.states[i],e0*e0.dag()) for i in range(len(sol_d.states))]

    # # print(sol_d.states[1])
    # expect_sz=[expect(sol_d.states[i],sz) for i in range(len(sol_d.states))]
    # expect_sx=[expect(sol_d.states[i],sx) for i in range(len(sol_d.states))]
    # expect_sy=[expect(sol_d.states[i],sy) for i in range(len(sol_d.states))]
    # # vec=[[expect(sol_d.states[i],sz),expect(sol_d.states[i],sz),expect(sol_d.states[i],sz)] for i in range(len(sol_d.states))]
    # vec=[expect_sx,expect_sy,expect_sz]
    fg_d,arg,eigenvals_t_d = fases(sol_d)
    ax_fg[j].plot(t/T,fg_d/np.pi,color='black')
    ax_concu[j].plot(t/T,concu,color='black')
    # ax[j].plot(t/T,prob_g0,color='blue')
    ax[j].plot(t/T,[sol_d.states[i][0][0] for i in range(len(sol_d.states))],color=colors[1],label='e0')
    ax[j].plot(t/T,[sol_d.states[i][2][2] for i in range(len(sol_d.states))],color=colors[0],label='g0')
    ax[j].plot(t/T,[sol_d.states[i][3][3] for i in range(len(sol_d.states))],color=colors[3],label='g1')
    ax[j].plot(t/T,[np.abs(sol_d.states[i][0][3]) for i in range(len(sol_d.states))],color=colors[2],linestyle='dashed',label='$C_{e0,g1}')
    
    # ax[j].plot(t/T,prob_g1,color='red')
    # ax[j].plot(t/T,prob_e0,color='green')
    # esferas[j].add_points(vec)
    # plt.plot(t/T,fg_d/np.pi)
ax1.set_xlim(0,25)
ax2.set_xlim(0,25)
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
ax1.set_title('(a) WC',size=20)
ax2.set_title('(b) SC',size=20)
ax1_concu.set_ylim(0,1)
ax2_concu.set_ylim(0,1)
ax1_fg.set_title('(a) SC',size=20)
ax2_fg.set_title('(b) WC',size=20)
ax1_fg.set_ylabel('$\phi_g/\pi$',size=20)
ax1_concu.set_ylabel('$C_{AF}$',size=20)
ax2_fg.set_ylabel('$\phi_g/\pi$',size=20)
ax2_concu.set_ylabel('$C_{AF}$',size=20)
ax1_concu.set_xlabel('$t/T$',size=20)
ax2_concu.set_xlabel('$t/T$',size=20)
plt.grid()
plt.show()
# esfera1.render()
# esfera1.show()
# esfera2.render()
# esfera2.show()