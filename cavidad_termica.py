from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jcm_lib as jcm
import os

script_path= os.path.dirname(__file__)

N_c=10


#Matriz de cambio de base
M=np.zeros((4*N_c,4*N_c))
M[0,3*N_c]=1
M[1,3*N_c+1]=1
M[2,N_c]=1/np.sqrt(2)
M[2,2*N_c]=1/np.sqrt(2)
M[3,N_c]=1/np.sqrt(2)
M[3,2*N_c]=-1/np.sqrt(2)

for ii in range(1,N_c-1):
    M[4*ii,3*N_c+1+ii]=1
for ii in range(1,N_c):
    M[4*ii+1,N_c+ii]=1/np.sqrt(2)
    M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
for ii in range(1,N_c):
    M[4*ii+2,ii-1]=1

 #Esta columna deberia pertenecer al gg,n+1, pero no existe asi que la matriz tiene 0's en esta fila. Para poder invertirla le ponemos un 1 en el estado een, para que el een se mapee al een, y listo. El estado gg,N+1 y gg,N+1 no estan disponibles

ee=basis([2,2],[0,0])
eg=basis([2,2],[0,1])
ge=basis([2,2],[1,0])
gg=basis([2,2],[1,1])

n=tensor(qeye(2),qeye(2),num(N_c)).transform(M)
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)]))).transform(M)
a=tensor(qeye(2),qeye(2),destroy(N_c)).transform(M)
sm1=tensor(sigmam(),qeye(2),qeye(N_c)).transform(M)
sp1=tensor(sigmap(),qeye(2),qeye(N_c)).transform(M)
sz1=tensor(sigmaz(),qeye(2),qeye(N_c)).transform(M)
sx1=tensor(sigmax(),qeye(2),qeye(N_c)).transform(M)
sm2=tensor(qeye(2),sigmam(),qeye(N_c)).transform(M)
sp2=tensor(qeye(2),sigmap(),qeye(N_c)).transform(M)
sz2=tensor(qeye(2),sigmaz(),qeye(N_c)).transform(M)
sx2=tensor(qeye(2),sigmax(),qeye(N_c)).transform(M)

# def beta_n(n_:int,k:float,J:float,x:float):
#     return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

# def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

# def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

# def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

# def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

# def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

# def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
#     return 2*np.sqrt(-2*Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)
# print(1/2*(eg+ge)*(eg+ge).dag())

##### DIFERENTES ESTADOS INICIALES ######
#CAVIDAD EN FOCK CON NUMERO BIEN DEFINIDO
rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),basis(N_c,0)*basis(N_c,0).dag()).transform(M) 

#CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO
rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0)+basis(N_c,1)))
# rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0))+tensor(ee,basis(N_c,1)))

#CAVIDAD TERMICA
rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),thermal_dm(N_c,2)).transform(M) 

#CAVIDAD COHERENTE
rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),coherent_dm(N_c,2)).transform(M) 




# with open("output.txt", "a") as file_object:
#     print("-------------------------------------------------------------------------------------------------------------------------------------------", file=file_object)
#     print(f"TERMINAL {script_path} cavidad_termica.py", file=file_object)
#     print("rho_c", file=file_object)
#     print(rho_c, file=file_object)
#     print("rho_0", file=file_object)
#     print(rho_0, file=file_object)
    
w0=1
g=0.001*w0


gamma=0*g       #.1*g
p=0.05*gamma

x=0         #1*g va en orden ascendiente
d=1*g        #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g

steps=2000

# T=2*np.pi/omega_general(1,1,d,g,k,J,x)
t_final=100/g

modelo='SB'

'''##########---Hamiltoniano---##########'''
if modelo=='TCM' or modelo=='1':
    #Hamiltoniano de TC
    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
elif modelo=='RABI' or modelo=='2':
    #Hamiltoniano de Rabi
    H=x*n2 + d/2*(sz1+sz2) + g*(sx1+sx2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
elif modelo=='SpinBoson' or modelo=='SB' or modelo=='3':
    #Hamiltoniano de "spin-boson"
    H=x*n2 + d/2*(sz1+sz2) + g*(sz1+sz2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
else:
    print('Este Modelo no existe. Modelo default es TCM.')
    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2]

sol=mesolve(H,rho_0,t,c_ops=l_ops)

fg_total,arg_tot,eigenvals_tot_t=jcm.fases_mixta(sol)

atoms_states=np.empty_like(sol.states)
for j in range(len(sol.states)):
    atoms_states[j]=sol.states[j].ptrace([0,1])  

fg_spins,arg_spins,eigenvals_t=jcm.fases(atoms_states)

concu_ab=jcm.concurrence(atoms_states)

color='black'
fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')
ax_fg=fig_fg.add_subplot()
ax_fg.set_xlim(0,t_final*g)
ax_fg.plot(t*g,fg_total/np.pi,color=color)

fig_concu=plt.figure(figsize=(8,6))
fig_concu.suptitle('CONCU')
ax_concu=fig_concu.add_subplot()
ax_concu.set_xlim(0,g*t_final)
ax_concu.plot(g*t,concu_ab,color=color)

fig_fg_spins=plt.figure(figsize=(8,6))
fig_fg_spins.suptitle('FG SPINS')
ax_fg_spins=fig_fg_spins.add_subplot()
ax_fg_spins.set_xlim(0,g*t_final)
ax_fg_spins.plot(g*t,fg_spins/np.pi,color=color)



def anim_hinton(rho):
    fig,ax=plt.subplots()
    fig_r,ax_r=plt.subplots()
    def init():
        """Initial drawing of the Hinton plot"""
        ax.clear()
        hinton(rho[0], ax=ax)
        ax.set_title("Frame 0")
        return ax,

    def init_r():
        ax_r.clear()
        hinton(rho[0].ptrace([0,1]),ax=ax_r)
        ax_r.set_title("Frame 0")
        return ax_r,

    def update(frame:int):
        """Update the Hinton plot for each frame"""
        ax.clear()
        hinton(rho[frame],x_basis=[],y_basis=[], ax=ax,colorbar=False)
        ax.set_title(f"Frame {frame}")
        return ax,

    def update_r(frame:int):
        ax_r.clear()
        hinton(rho[frame].ptrace([0,1]),x_basis=[],y_basis=[], ax=ax_r,colorbar=False)
        ax_r.set_title(f"Frame {frame}")
        return ax_r

    # Create animation
    anim_h= FuncAnimation(fig, update, frames=len(rho), init_func=init, blit=False, repeat=True)
    anim_h_r= FuncAnimation(fig_r, update_r, frames=len(rho), init_func=init_r, blit=False, repeat=True)
    # plt.show()
    return anim_h,anim_h_r


anim_h,anim_h_r=anim_hinton(sol.states)
anim_h.save(f'hinton/hinton {modelo} tot.mp4','ffmpeg',5)
anim_h_r.save(f'hinton/hinton {modelo} spins.mp4','ffmpeg',5)
# fig0=plt.figure()
# ax0=fig0.add_subplot()
# hinton(sol.states[1],x_basis=[],y_basis=[], ax=ax0,colorbar=False)
# Show the animation

# plt.show()
