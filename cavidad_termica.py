from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jcm_lib as jcm
import os
import time

script_path= os.path.dirname(__file__)

N_c=11
steps=500
g_t=5

cond_inic=[0] #,1,2,3] #[0,1,2,3]
modelos=['TCM','SB'] #['RABI','TCM','SB']

w0=1
g=0.001*w0


gamma=0.1*g       #.1*g
p=0.05*gamma

x=0         #1*g va en orden ascendiente
d=0*g        #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g




#Matriz de cambio de base
# M=np.eye(4*N_c)
M=np.zeros((4*N_c,4*N_c))
M[0,3*N_c]=1
M[1,3*N_c+1]=1
M[2,N_c]=1/np.sqrt(2)
M[2,2*N_c]=1/np.sqrt(2)
M[3,N_c]=1/np.sqrt(2)
M[3,2*N_c]=-1/np.sqrt(2)

for ii in range(1,N_c-1):
    M[4*ii,3*N_c+1+ii]=1
for ii in range(1,N_c-1):
    M[4*ii+1,N_c+ii]=1/np.sqrt(2)
    M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
for ii in range(1,N_c-1):
    M[4*ii+2,ii-1]=1

ee=basis([2,2],[0,0])
eg=basis([2,2],[0,1])
ge=basis([2,2],[1,0])
gg=basis([2,2],[1,1])

n=tensor(qeye(2),qeye(2),num(N_c))
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
a=tensor(qeye(2),qeye(2),destroy(N_c))
sm1=tensor(sigmam(),qeye(2),qeye(N_c))
sp1=tensor(sigmap(),qeye(2),qeye(N_c))
sz1=tensor(sigmaz(),qeye(2),qeye(N_c))
sx1=tensor(sigmax(),qeye(2),qeye(N_c))
sm2=tensor(qeye(2),sigmam(),qeye(N_c))
sp2=tensor(qeye(2),sigmap(),qeye(N_c))
sz2=tensor(qeye(2),sigmaz(),qeye(N_c))
sx2=tensor(qeye(2),sigmax(),qeye(N_c))

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

for ci in cond_inic:
    if ci==0:
        #CAVIDAD EN FOCK CON NUMERO BIEN DEFINIDO
        fotones=2
        eef=tensor(ee,basis(N_c,fotones-2))
        ggf=tensor(gg,basis(N_c,fotones))
        rho_0=(eef+ggf).unit()*(eef+ggf).unit().dag()
        # rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),basis(N_c,fotones)*basis(N_c,fotones).dag())
    elif ci==1:
        #CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO
        rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0)+basis(N_c,1)))
        # rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0))+tensor(ee,basis(N_c,1)))
    elif ci==2:
        #CAVIDAD TERMICA
        rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),thermal_dm(N_c,2)) 
    elif ci==3:
        #CAVIDAD COHERENTE
        rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),coherent_dm(N_c,2)) 
    else:
        print('porfavor elegir una condicion inicial que este suporteada. Por default ci=0')
        ci=0

    t_final=g_t/g

    for modelo in modelos:
        '''##########---Hamiltoniano---##########'''
        if modelo=='TCM' or modelo=='1':
            #Hamiltoniano de TC
            H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
        elif modelo=='RABI' or modelo=='2':
            #Hamiltoniano de Rabi
            H=x*n2 + d/2*(sz1+sz2) + g*(sx1+sx2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
        elif modelo=='SpinBoson' or modelo=='SB' or modelo=='3':
            #Hamiltoniano de "spin-boson"
            H=d/2*(sz1+sz2) + g*(sz1+sz2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
        else:
            print('Este Modelo no existe. Modelo default es TCM.')
            H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
        print(modelo,' ci:',ci )
        t_0=time.time()

        t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2]

        sol=mesolve(H,rho_0,t,c_ops=l_ops)

        t_i_cb=time.time()
        for i in range(len(sol.states)):
            sol.states[i]=sol.states[i].transform(M)
            sol.states[i].full()[np.abs(sol.states[i].full()) <=1e-8]=0 


        t_f_cb=time.time()
        print(t_f_cb-t_i_cb,'s tiempo de cambio de base')

        fg_total,arg_tot,eigenvals_tot_t=jcm.fases_mixta(sol)

        atoms_states=np.empty_like(sol.states)
        for j in range(len(sol.states)):
            atoms_states[j]=sol.states[j].ptrace([0,1])  

        fg_spins,arg_spins,eigenvals_t=jcm.fases(atoms_states)

        concu_ab=jcm.concurrence(atoms_states)

        color='black'
        def figura_plot(id:str,x:list,y:list,color:list=['black'],figsize:list=(8,6)):
            fig=plt.figure(id,figsize=figsize)
            fig.suptitle(id)
            ax=fig.add_subplot()
            ax.set_xlim(0,t_final*g)
            ax.plot(x,y,color=color[0])

        figura_plot('fg',g*t,fg_total/np.pi)
        plt.figure('fg').savefig(f'fg/fg {modelo} {ci} plot.png')
        
        figura_plot('concu',g*t,concu_ab)
        plt.figure('concu').savefig(f'fg/concu {modelo} {ci} plot.png')

        figura_plot('fg spin',g*t,fg_spins/np.pi)
        plt.figure('fg spin').savefig(f'fg/fg spin {modelo} {ci} plot.png')

        plt.figure('fg',clear=True)
        plt.figure('concu',clear=True)
        plt.figure('fg spin',clear=True)
        def anim_hinton(rho):
            fig=plt.figure('hinton tot',figsize=(8,6))
            ax=fig.add_subplot()
            fig_r=plt.figure('hinton r',figsize=(8,6))
            ax_r=fig_r.subplots()
            def init():
                """Initial drawing of the Hinton plot"""
                ax.clear()
                hinton(rho[0], ax=ax,color_style="phase")
                ax.set_title("Frame 0")
                return ax,

            def init_r():
                ax_r.clear()
                hinton(rho[0].ptrace([0,1]),ax=ax_r,color_style="phase")
                ax_r.set_title("Frame 0")
                return ax_r,

            def update(frame:int):
                """Update the Hinton plot for each frame"""
                ax.clear()
                hinton(rho[frame],x_basis=[],y_basis=[], ax=ax,colorbar=False,color_style="phase")
                ax.set_title(f"Frame {frame}")
                return ax,

            def update_r(frame:int):
                ax_r.clear()
                hinton(rho[frame].ptrace([0,1]),x_basis=[],y_basis=[], ax=ax_r,colorbar=False,color_style="phase")
                ax_r.set_title(f"Frame {frame}")
                return ax_r
            t_i_anim=time.time()
            # Create animation
            anim_h= FuncAnimation(fig, update, frames=len(rho), init_func=init, blit=False, repeat=True,cache_frame_data=False)
            anim_h_r= FuncAnimation(fig_r, update_r, frames=len(rho), init_func=init_r, blit=False, repeat=True,cache_frame_data=False)
            t_f_anim=time.time()
            print(t_f_anim-t_i_anim,'s tiempo de procesamiento de hinton')
            # plt.show()
            return anim_h,anim_h_r

        t_in=time.time()
        print(t_in-t_0,'s tiempo de computo de simulacion')
        anim_h,anim_h_r=anim_hinton(sol.states)
        t_i_guardado=time.time()
        if gamma==0:
            anim_h.save(f'hinton/hinton uni {modelo} {ci} d={d/g}g tot.mp4','ffmpeg',5)
            print('fin guardado ginton tot')
            anim_h_r.save(f'hinton/hinton uni {modelo} {ci} d={d/g}g spins.mp4','ffmpeg',5)
            print('fin guardado hinton spins')
        elif gamma>0:
            anim_h.save(f'hinton/hinton dis {modelo} {ci} d={d/g}g tot.mp4','ffmpeg',5)
            print('fin guardado ginton tot')
            anim_h_r.save(f'hinton/hinton dis {modelo} {ci} d={d/g}g spins.mp4','ffmpeg',5)
            print('fin guardado hinton spins')
        t_fin=time.time()
        print(t_fin-t_i_guardado,'s de guardado de hinton')


        plt.close()