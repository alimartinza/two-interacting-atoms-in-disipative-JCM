from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence_ali
import matplotlib as mpl
from entrelazamiento_lib import negativity_hor
import os 

#ES EL HERMANO DE LEEDOR_JCM.PY, LA IDEA ES HACER SIMULACIONES LARGAS ACA Y DESPUES LEERLAS CON LEEDOR.
# LAS SIMULACIONES QUE HACEMOS ACA SON DE JCM DE 1 ATOMO Y HAY VARIAS COSAS
# 1. SUPERFICIE DE FG. BARREMOS EN DELTA Y HACEMOS UN PLOT DE LA FG
# 2. TRAYECTORIAS DE BLOCH
# 3. GRAFICOS TRANQUI PARA MIRAR CASOS PARTICULARES DE FG
# 4. BLOCH DISIP VS UNIT
# 5. BLOCH BARRIDA EN CONDICION INICIAL Y ENTRELAZAMIENTO DE CADA EVOLUCION DE LA CONDICION INICIAL CON HEATMAP

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



def omega_n(n_:int,delta:float):
    return np.sqrt(delta**2+4*g**2*n_)

def cos_theta_n(n_:int,delta:float):
    return np.sqrt((omega_n(n_,delta)+delta)/(2*omega_n(n_,delta)))

def sin_theta_n(n_:int,delta:float):
    return np.sqrt((omega_n(n_,delta)-delta)/(2*omega_n(n_,delta)))

def pr(estado):
    return estado.unit()*estado.unit().dag()

def omega_n(n_:int,delta:float,chi:float):
    return np.sqrt((delta-chi*(2*n_-1))**2+4*g**2*n_)

def vectorBloch(v1,v2,sol_states,steps,ciclos_bloch,T,t_final,points):
    sz_1=pr(v1)-pr(v2)
    sx_1=v1*v2.dag()+v2*v1.dag()
    sy_1=-1j*v1*v2.dag()+1j*v2*v1.dag()

    expect_sx_1=[expect(sx_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    expect_sy_1=[expect(sy_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    expect_sz_1=[expect(sz_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    return [expect_sx_1,expect_sy_1,expect_sz_1]


'''--------------- CORRIDA Y BARRIDO EN DELTA --------------'''
#LA IDEA ES BARRER EN DELTA Y MIRAR PHI_D-PHI_U EN FUNCION DEL TIEMPO, Y MARCAR EN EL PLOT 3D CUANDO LOS AUTOVALORES SON CERO Y 
# CUANDO LA NEGATIVITY REVIVE (A TIEMPOS LARGOS).
#SUPERFICIE 3D WIREFRAME EJEX=DELTA EJEY=TIEMPO EJEZ=DELTAFG


# steps=160000

# ciclos_bloch=40
# points=ciclos_bloch*50
# for p in [0.1*0.1*g]:
#     for gamma in [0.1*g]:
#         for x in [0*g]:
#             for tita in [0.*np.pi]:
#                 delta_array=[0.0000001*g]
#                 omega=np.sqrt(4*g**2+(0-x)**2)
#                 '''---Simulacion numerica---'''
#                 T=2*np.pi/omega
#                 t_final=40*T

#                 t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#                 fg_delta=np.zeros((len(delta_array),steps))
#                 fg_u_delta=np.zeros((len(delta_array),steps))
#                 fg_d_delta=np.zeros((len(delta_array),steps))
#                 # print(fg_delta[0])
#                 N_u_delta=np.zeros((len(delta_array),steps))
#                 N_d_delta=np.zeros((len(delta_array),steps))



#                 eigvals_death_t=np.full(len(delta_array),-1)
#                 # eigvals_death_z=np.full(len(delta_array),0)

#                 negativity_revival_t=np.full(len(delta_array),-1)
#                 negativity_revival_z=np.full(len(delta_array),0)

#                 for i_delta,delta in enumerate(delta_array):
#                     print(f'delta #{i_delta}/{len(delta_array)}')
#                     # psi0=(e0+(1+1j)*g1).unit()#(np.sqrt(2+np.sqrt(2))/2*e0+1j*np.sqrt(2-np.sqrt(2))/2*g1).unit()
                    
#                     phi=0
#                     psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#                     H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)

#                     l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
                    
#                     sol_u=mesolve(H,psi0,t)
#                     sol_d=mesolve(H,psi0,t,c_ops=l_ops)

#                     fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#                     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#                     fg_delta[i_delta]=fg_d-fg_u
#                     fg_d_delta[i_delta]=fg_d
#                     fg_u_delta[i_delta]=fg_u

#                     N_u=np.array([negativity_hor(sol_u.states[i],[0,1]) for i in range(len(sol_u.states))])
#                     N_d=np.array([negativity_hor(sol_d.states[i],[0,1]) for i in range(len(sol_d.states))])

#                     # for i_find in range(0,band-1,steps):
#                     #     #if np.array_equiv(N_d[i_find-band:i_find],np.zeros(band)) and np.count_nonzero(N_d[i_find+1:i_find+1+band])>percent*band:
#                     #     if N_d[i_find]<1e-6 and N_d[i_find+1]>1e-4:
#                     #         negativity_revival_t[i_delta]=t[i_find]
#                     #         negativity_revival_z[i_delta]=fg_delta[i_delta][i_find]
#                     #         print(f'i_find = {i_find}')
#                     #         break 

#                     N_u_delta[i_delta]=N_u
#                     N_d_delta[i_delta]=N_d

#                     mask = np.full(steps, True)
#                     mask[:11] = False  # Skip the first element
                    
#                     # Check if eigenvals_t_d has the expected structure
#                     if hasattr(eigenvals_t_d, 'shape') and len(eigenvals_t_d.shape) > 1:
#                         for i_ev in range(1, eigenvals_t_d.shape[1]):  # Iterate over columns (time steps)
#                             mask_step = eigenvals_t_d[:, i_ev] < 1e-5
#                             mask = mask & mask_step  # Use bitwise AND for numpy arrays
                    
#                     # Find first True using numpy methods
#                     true_indices = np.where(mask)[0]
#                     if len(true_indices) > 0:
#                         first_true_index = true_indices[0]
#                         eigvals_death_t[i_delta] = t[first_true_index]
#                         # eigvals_death_z[i_delta] = fg_delta[i_delta][first_true_index]
#                     else:
#                         eigvals_death_t[i_delta] = -1
#                         # eigvals_death_z[i_delta] = -1
                
               
#                 fig_fg=plt.figure()
#                 ax_fg=fig_fg.add_subplot(projection='3d')
#                 dELTA, tT = np.meshgrid(delta_array, t/T,indexing='ij')
#                 ax_fg.plot_wireframe(dELTA,tT,fg_delta,cstride=len(delta_array))
#                 # ax_fg.scatter(delta_array,eigvals_death_t/T,eigvals_death_z,color='red')
#                 # ax_fg.scatter(delta_array,negativity_revival_t/T,negativity_revival_z,color='green')
#                 ax_fg.set_xlabel(r'$\Delta$')
#                 ax_fg.set_ylabel(r'$t/T$')
#                 ax_fg.set_zlabel(r'$\delta \phi$')
#                 plt.show()
#                 



                # plt.show()
                # np.savetxt(f'datajcm/evalst death x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',eigvals_death_t)
                # np.savetxt(f'datajcm/fgu x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',fg_u_delta)
                # np.savetxt(f'datajcm/fgd x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',fg_d_delta)
                # np.savetxt(f'datajcm/Nu x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',N_u_delta)
                # np.savetxt(f'datajcm/Nd x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',N_d_delta)

                # np.savetxt(f'datajcm/vBu_x x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_u_delta[0])
                # np.savetxt(f'datajcm/vBu_y x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_u_delta[1])
                # np.savetxt(f'datajcm/vBu_z x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_u_delta[2])

                # np.savetxt(f'datajcm/vBd_x x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_d_delta[0])
                # np.savetxt(f'datajcm/vBd_y x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_d_delta[1])
                # np.savetxt(f'datajcm/vBd_z x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_d_delta[2])
                
                # np.savetxt(f'datajcm/vBev_x x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_eigenvec_delta[0])
                # np.savetxt(f'datajcm/vBev_y x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_eigenvec_delta[1])
                # np.savetxt(f'datajcm/vBev_z x{x/g} ga{gamma/g} p{p/g:.3f} tita{tita/np.pi}.txt',vBloch_eigenvec_delta[2])


# plt.show()

# fig_N=plt.figure()
# ax_N=fig_N.add_subplot(projection='3d')
# ax_N.plot_surface(dELTA,gT,N_d_delta,cstride=len(delta_array))
# ax_N.set_xlabel(r'$\Delta$')
# ax_N.set_ylabel(r'$gt$')
# ax_N.set_zlabel(r'$N_d$')
# plt.show()

'''----BLOCH---'''
#ESFERA DE BLOCH. LA IDEA ES ELEGIR ALGUNAS COMBINACIONES DE PARAMETROS PARA COMPARAR LAS TRAYECTORIAS 
#PARA LOS DIFERENTES CASOS.

# ciclos_bloch=40
# steps=3000*ciclos_bloch

# points=ciclos_bloch*50
# for p in [0.1*0.1*g]:
#     for gamma in [0.1*g]:
#         for x in [0*g]:
#             for tita in [0.*np.pi]:
#                 delta_array=[0.0000001*g]
#                 omega=np.sqrt(4*g**2+(0-x)**2)
#                 '''---Simulacion numerica---'''
#                 T=2*np.pi/omega
#                 t_final=40*T

#                 t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#                 fg_delta=np.zeros((len(delta_array),steps))
#                 fg_u_delta=np.zeros((len(delta_array),steps))
#                 fg_d_delta=np.zeros((len(delta_array),steps))

#                 vBloch_u_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#                 vBloch_d_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#                 vBloch_eigenvec_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))


#                 for i_delta,delta in enumerate(delta_array):
#                     print(f'delta #{i_delta}/{len(delta_array)}')
#                     # psi0=(e0+(1+1j)*g1).unit()#(np.sqrt(2+np.sqrt(2))/2*e0+1j*np.sqrt(2-np.sqrt(2))/2*g1).unit()
                    
#                     phi=0
#                     psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#                     H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)

#                     l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
                    
#                     sol_u=mesolve(H,psi0,t)
#                     sol_d=mesolve(H,psi0,t,c_ops=l_ops)

#                     fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#                     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#                     fg_delta[i_delta]=fg_d-fg_u
#                     fg_d_delta[i_delta]=fg_d
#                     fg_u_delta[i_delta]=fg_u
#                 vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
#                 esfera1=Bloch()
#                 esfera1.make_sphere()
#                 colors=[mpl.colormaps['viridis'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['winter'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['magma'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))]

#                 esfera1.add_points(vBloch_u,'m',colors='black')
#                 esfera1.add_points(vBloch_eigevec,'m',colors=colors[1])
#                 esfera1.add_points(vBloch_d,'m',colors=colors[2])
#                 esfera1.render()
#                 # # esfera.save('bloch berry.png')
#                 esfera1.show()
#                 plt.show()


'''----FG con pocos delta para analizar casos------'''

# fig_tau=plt.figure(figsize=(8,6))
# ax_tau=fig_tau.add_subplot()
# fig_fg=plt.figure(figsize=(8,6))
# ax_fg=fig_fg.add_subplot()
# esfera1=Bloch()
# esfera1.make_sphere()
# w_0=1
# g=0.01*w_0

# colors=['black','red','blue','green']
# i_color=0
# steps=3000*10
# # tita=0
# ciclos_bloch=10
# points=ciclos_bloch*50
# for p in [0.1*0.1*g]:
#     for gamma in [0.1*g]:
#         for x in [0*g]:
#             # for steps in [160,1600]:
#             delta_array=[2*g]
#             omega=np.sqrt(4*g**2+(delta_array[0]-x)**2)
#             '''---Simulacion numerica---'''
#             T=2*np.pi/omega
#             t_final=10*T
             
#             t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#             fg_delta=np.zeros((len(delta_array),steps))
#             fg_u_delta=np.zeros((len(delta_array),steps))
#             fg_d_delta=np.zeros((len(delta_array),steps))
#             # print(fg_delta[0])
#             N_u_delta=np.zeros((len(delta_array),steps))
#             N_d_delta=np.zeros((len(delta_array),steps))

#             vBloch_u_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#             vBloch_d_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#             vBloch_eigenvec_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))

#             eigvals_death_t=np.full(len(delta_array),-1)
#             # eigvals_death_z=np.full(len(delta_array),0)
#             # negativity_revival_t=np.full(len(delta_array),-1)
#             # negativity_revival_z=np.full(len(delta_array),0)

#             for i_delta,delta in enumerate(delta_array):
#                 tita_rob=np.arctan2(delta-x,-2*g)
#                 tita=tita_rob
#                 print(f'{tita/np.pi}pi') 
#                 phi=0
#                 psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#                 H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
#                 # print(psi0)
#                 # print(H*psi0)
#                 l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
                
#                 sol_u=mesolve(H,psi0,t)
#                 sol_d=mesolve(H,psi0,t,c_ops=l_ops)

#                 fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#                 fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#                 fg_delta[i_delta]=fg_d-fg_u
#                 fg_d_delta[i_delta]=fg_d
#                 fg_u_delta[i_delta]=fg_u
#                 vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
#                 esfera1.add_points(vBloch_u,'m',colors='black')
#                 esfera1.add_points(vBloch_eigevec,'m',colors='red')
#                 esfera1.add_points(vBloch_d,'m',colors='yellow')

#                 label=fr'$\gamma={gamma/g :.2f}g, p={p/g :.2f}g$'
#                 dfg=fg_d-fg_u


#                 ax_fg.plot(t/T,fg_u/np.pi,color=colors[i_color],label=f'u')
#                 ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_color],linestyle='dashed',label=f'd')
#                 # ax_fg.scatter(t/T,dfg,color=colors[i_color],label=f'd-u')
#                 # ax_fg.vlines(t_saltos,-2,2)
#                 # plt.show()

#                 # index_saltos=[]
#                 # t_saltos=[]

#                 # for i_tau in range(len(dfg)-1):
#                 #     if np.abs(dfg[i_tau+1]-dfg[i_tau])>3:
#                 #         index_saltos.append(i_tau)
#                 #         t_saltos.append(t[i_tau]/T)
#                 # dif_tiempo_salto=[]
#                 # for i in range(0,len(t_saltos)-2,2):
#                 #     dif_tiempo_salto.append(t_saltos[i+1]-t_saltos[i])
#                 # ax_tau.plot(dif_tiempo_salto,color=colors[i_color])
#                 # ax_tau.scatter(range(len(dif_tiempo_salto)),dif_tiempo_salto,color=colors[i_color],label=label)
#                 # i_color+=1
# esfera1.render()
# esfera1.show()
# # ax_tau.legend()

# ax_fg.legend()

# plt.show()

'''-------------------------- GRAFICOS FUNCIONABLES --------------------------'''

# gamma=0.1*g
# p=0.1*0.1*g

# points=15000
# x=0*g
# delta=0*g
# # psi0=(e0+(1+1j)*g1).unit()#(np.sqrt(2+np.sqrt(2))/2*e0+1j*np.sqrt(2-np.sqrt(2))/2*g1).unit()
# tita=0
# phi=0
# psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
# H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
# omega=np.sqrt(4*g**2+(delta-x)**2)
# '''---Simulacion numerica---'''
# T=2*np.pi/omega
# t_final=70*T
# steps=15000
# ciclos_bloch=70
# colors=[mpl.colormaps['viridis'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['winter'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['magma'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))]

'''-------------------------    BLOCH UNIT PUNTERO DISIP FG --------------------------'''

# fig_e=plt.figure(figsize=(8,6))

# ax_n=fig_e.add_subplot(321)
# ax_n_zoom=fig_e.add_subplot(322)
# ax_n.set_xlabel('$t/T$')
# ax_n.set_ylabel(r'$\mathcal{N}(\rho)$')
# # ax_e.set_ylabel(r'$E(\rho)$')
# ax_fg=fig_e.add_subplot(323,sharex=ax_n)
# ax_fg_zoom=fig_e.add_subplot(324,sharex=ax_n_zoom)

# ax_e=fig_e.add_subplot(325,sharex=ax_fg)
# ax_e_zoom=fig_e.add_subplot(326,sharex=ax_fg_zoom)
# ax_e.set_xlabel('$t/T$')
# ax_e.set_ylabel(r'$E(\rho)$')


# colors_fg=['blue','black','red']
# labels_fg=['u','d','d+']

# l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
# t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 


# sol_u=mesolve(H,psi0,t)
# sol_d=mesolve(H,psi0,t,c_ops=l_ops)

# fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
# fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

# N_u=np.array([negativity_hor(sol_u.states[i],[0,1]) for i in range(len(sol_u.states))])
# N_d=np.array([negativity_hor(sol_d.states[i],[0,1]) for i in range(len(sol_d.states))])

# # C_u=concurrence_ali(sol_u.states)
# # C_d=concurrence_ali(sol_d.states)

# vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
# vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
# vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
# esfera1=Bloch()
# esfera1.make_sphere()

# esfera1.add_points(vBloch_u,'m',colors='black')
# esfera1.add_points(vBloch_eigevec,'m',colors=colors[1])
# esfera1.add_points(vBloch_d,'m',colors=colors[2])

# # vBloch_u=vectorBloch(e1,g2,sol_u.states,steps,ciclos_bloch,T,t_final,points)
# # vBloch_eigevec=vectorBloch(e1,g2,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
# # vBloch_d=vectorBloch(e1,g2,sol_d.states,steps,ciclos_bloch,T,t_final,points)
# esfera1.render()
# # esfera.save('bloch berry.png')
# esfera1.show()

# # esfera2=Bloch()
# # esfera2.make_sphere()

# # esfera2.add_points(vBloch_u,'m',colors='black')
# # esfera2.add_points(vBloch_eigevec,'m',colors=colors[1])
# # esfera2.add_points(vBloch_d,'m',colors=colors[2])
# # esfera2.render()
# # # esfera.save('bloch berry.png')
# # esfera2.show()

# zoom_steps=steps
# colors_e=mpl.colormaps['hot'](np.linspace(0,1,len(eigenvals_t_d[0])))
# for i1 in range(len(eigenvals_t_d[0])): 
#     if i1==2:
#         max_eig2=np.max(eigenvals_t_d[:,i1])
#         eigenvals_t_d[:,i1]=eigenvals_t_d[:,i1]/max_eig2
#         ax_e.text(0.6*t_final/T,0.75,"x{0:.2E}".format(max_eig2),color=colors_e[i1])
#     ax_e.plot(t/T,eigenvals_t_d[:,i1],color=colors_e[i1])
#     ax_e_zoom.plot(t[:zoom_steps]/T,eigenvals_t_d[:zoom_steps,i1],color=colors_e[i1])

# ax_n.plot(t/T,N_u,color='red',linestyle='dashed',label='N_u')
# ax_n.plot(t/T,N_d,color='green',linestyle='dashed',label='N_d')

# ax_n_zoom.plot(t[:zoom_steps]/T,N_u[:zoom_steps],color='red',linestyle='dashed',label='N_u')
# ax_n_zoom.plot(t[:zoom_steps]/T,N_d[:zoom_steps],color='green',linestyle='dashed',label='N_d')

# ax_fg.plot(t/T,fg_u,color=colors_fg[0],label=labels_fg[0])
# ax_fg.plot(t/T,fg_d,color=colors_fg[1],label=labels_fg[1])

# ax_fg_zoom.plot(t[:zoom_steps]/T,fg_u[:zoom_steps],color=colors_fg[0],label=labels_fg[0])
# ax_fg_zoom.plot(t[:zoom_steps]/T,fg_d[:zoom_steps],color=colors_fg[1],label=labels_fg[1])

# ax_fg.set_xlabel('$t/T$')
# ax_fg.set_ylabel(r'$\phi_g$')
# ax_fg.legend()
# ax_e.legend()
# plt.show()

'''------------------   CONDICIONES INICIALES TITA  ---------------------------------------------------------'''
#ESFERA DE BLOCH Y HEATPLOT DE NEGATIVIDAD. BARREMOS EN UN ANGULO TITA (DE CONDICION INICIAL). MIRAMOS 1 CICLO DE EVOLUCION PARA CADA CONDICION INICIAL

# def heatplot(t,y,z_data:list,title:str,ylabel):
#     fig_u=plt.figure(figsize=(8,6))
#     fig_u.suptitle(title)
#     ax_u=fig_u.add_subplot()
#     ax_u.set_xlabel('$t/T$')
#     ax_u.set_ylabel(ylabel)
#     c0 = ax_u.pcolor(t/T, y, z_data, shading='auto', cmap='jet',vmin=0,vmax=0.5)
#     contour_u = ax_u.contourf(t/T, y, z_data,levels=[0,0.01],colors='black',linewidths=1)
#     ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
#     fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
#     # fig_u.savefig(rf'graficos\negativity\{psi0Name} {title} x={x/g}g k={k/g}g J={J/g}g neg delta dis.png')

# esfera=Bloch()
# esfera.make_sphere()


# #Parametros y Hamiltoniano

# gamma=0.1*g
# p=0.01*g#*0.1*g

# x=0*g
# delta=1*g

# H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
# omega=np.sqrt(4*g**2+(delta-x)**2)

# # Simulacion numerica
# num_ciclos=3
# T=2*np.pi/omega
# t_final=num_ciclos*T
# steps=3000*num_ciclos

# ciclos_bloch=2
# points=ciclos_bloch*50

# # Barrido de condiciones iniciales

# num_tita=15
# tita_eig=2*np.arctan2((-delta+x-np.sqrt(np.power((-delta+x),2)+4*g**2)),(2*g))
# tita_rob=np.arctan2(2*g,delta-x)+np.pi/2
# # tita_array=np.linspace(tita_eig,tita_eig+np.pi/2,num_tita)
# tita_array=np.linspace(0,np.pi,num_tita)
# colors=mpl.colormaps['plasma'](np.linspace(0,1,num_tita)) #colores para pintar en la esfera de bloch
# colors_map=mpl.colormaps['viridis'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))#,mpl.colormaps['winter'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))#,mpl.colormaps['magma'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))

# #definimos los arrays de negatividad

# N_u=np.zeros((num_tita,steps))
# N_d=np.zeros((num_tita,steps))

# print(f'{tita_eig/np.pi}pi')
# print(f'{tita_rob/np.pi}pi')
# #ahora hacemos el barrido
# #paara cada valor de tita hacemos una simulacion y calculamos 

# fig_fg_tita=plt.figure(figsize=(8,6))
# ax_fg_tita=fig_fg_tita.add_subplot()
# ax_fg_tita.set_xlabel(r'$t/T$')
# ax_fg_tita.set_ylabel(r'$\delta\phi_g/\pi$')

# phi=0
# # tita_array=[np.pi/4]
# for j,tita in enumerate(tita_array):
#     # phi=np.arcsin(0.5/np.sin(tita))
#     psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#     l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
#     t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#     sol_u=mesolve(H,psi0,t)
#     sol_d=mesolve(H,psi0,t,c_ops=l_ops)
    
#     fg_u,arg,eigenvals_t_d,psi_eig_u = fases(sol_u)
#     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)
#     ax_fg_tita.plot(t/T,fg_d/np.pi,color=colors[j],label=fr'$\theta={tita/np.pi}$',linestyle='dashed')
#     ax_fg_tita.plot(t/T,fg_u/np.pi,color=colors[j])
#     N_u[j]=np.array([negativity_hor(sol_u.states[i],[0,1]) for i in range(len(sol_u.states))])
#     N_d[j]=np.array([negativity_hor(sol_d.states[i],[0,1]) for i in range(len(sol_d.states))])

#     vBloch_tita=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
#     esfera.add_points(vBloch_tita,'m',colors=colors[j])

#     vBloch_tita=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
#     esfera.add_points(vBloch_tita,'m',colors=colors_map)
#     print(j)
# ax_fg_tita.legend()
# # print(N_d)
# heatplot(t,tita_array/np.pi,N_u,'Negativity unit',r"$\theta/\pi$")  
# heatplot(t,tita_array/np.pi,N_d,'Negativity disip',r"$\theta/\pi$")  
# # heatplot(t,tita_array/np.pi,fg_u/np.pi,r'$\phi_g$ uni',r"$\phi_g/\pi$")
# # heatplot(t,tita_array/np.pi,fg_d/np.pi,r'$\phi_g$ dis',r"$\phi_g/\pi$")
# # heatplot(t,tita_array/np.pi,(fg_u-fg_d)/np.pi,r'$\delta\phi$ dis',r"$\phi_g/\pi$")

# esfera.render()
# # esfera.save('bloch berry.png')
# esfera.show()
# plt.show()

'''------- BARRIDA DELTA CONDICION INICIAL PERPENDICULAR A LA DIRECCION DE ROTACION DEL HAMILTONIANO --------'''
#EL HAMILTONIANO DE 2X2 DEFINE UNA DIRECCION n=(g*sqrt(n),0,delta/2). PONEMOS LA CONDICION INICIAL QUE SEA 
#PERPENDICULAR A ESTA DIRECCION CALCULANDO EL ANGULO THETA DE LA DIRECCION n, SABIENDO QUE EL ANGULO POLAR
#phi=0 PORQUE EN n_y=0, DE ESTA MANERA LA TRAYECTORIA UNITARIA ES SIEMPRE POR UNA GEODESICA INDEPENDIENTEMENTE
#DE LA CONDICION DE RESONANCIA

def heatplot(t,y,z_data:list,title:str,ylabel):
    fig_u=plt.figure(figsize=(8,6))
    fig_u.suptitle(title)
    ax_u=fig_u.add_subplot()
    ax_u.set_xlabel('$t/T$')
    ax_u.set_ylabel(ylabel)
    c0 = ax_u.pcolor(t/T, y, z_data, shading='auto', cmap='jet',vmin=0,vmax=0.5)
    contour_u = ax_u.contourf(t/T, y, z_data,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    # fig_u.savefig(rf'graficos\negativity\{psi0Name} {title} x={x/g}g k={k/g}g J={J/g}g neg delta dis.png')


# fig_fg=plt.figure(figsize=(8,6))
# ax_fg=fig_fg.add_subplot()

w_0=1
g=0.01*w_0

gamma=0.1*g
p=0.01*g
x=0

steps=3000*9
delta_array=np.linspace(0,10*g,100)

fg_delta=np.zeros((len(delta_array),steps))
fg_u_delta=np.zeros((len(delta_array),steps))
fg_d_delta=np.zeros((len(delta_array),steps))
N_u_delta=np.zeros((len(delta_array),steps))
N_d_delta=np.zeros((len(delta_array),steps))
omega0=2*g
T0=2*np.pi/omega0

fig_fgs=plt.figure(figsize=(8,6))
ax_fgs=fig_fgs.add_subplot()
colors=mpl.colormaps['plasma'](np.linspace(0,1,len(delta_array)))

for i_delta,delta in enumerate(delta_array):
    print(i_delta)
    omega=np.sqrt(4*g**2+(delta-x)**2)
    T=2*np.pi/omega
    t_final=3*T
        
    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    tita_rob=np.arctan2(delta-x,-2*g)
    tita=tita_rob
    phi=0
    psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
    H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
    
    sol_u=mesolve(H,psi0,t)
    sol_d=mesolve(H,psi0,t,c_ops=l_ops)

    fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
    fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

    fg_delta[i_delta]=np.abs(fg_d)-np.abs(fg_u)
    fg_d_delta[i_delta]=np.abs(fg_d)
    fg_u_delta[i_delta]=np.abs(fg_u)

    N_u_delta[i_delta]=np.array([negativity_hor(sol_u.states[i],[0,1]) for i in range(len(sol_u.states))])
    N_d_delta[i_delta]=np.array([negativity_hor(sol_d.states[i],[0,1]) for i in range(len(sol_d.states))])
    ax_fgs.plot(t/T,fg_u,color=colors[i_delta],label=f'{delta}')
    ax_fgs.plot(t/T,fg_d,color=colors[i_delta],linestyle='dashed')
t=np.linspace(0,3*T0,steps)
heatplot(t/T0,delta_array/g,N_u_delta,'Negatividad u',r'$\Delta/g$')
heatplot(t/T0,delta_array/g,N_u_delta,'Negatividad d',r'$\Delta/g$')

heatplot(t/T0,delta_array/g,fg_delta/np.pi,r'$\delta \phi/\pi$',r'$\Delta/g$')

fig_rob=plt.figure(figsize=(8,6))
ax_rob=fig_rob.add_subplot()
ax_rob.scatter(delta_array/g,fg_delta[:,-1]/np.pi,color='red')
ax_rob.set_xlabel(r'$\Delta/g$')
ax_rob.set_ylabel(r'$\delta\phi/\pi$')

# ax_fgs.legend()

plt.show()