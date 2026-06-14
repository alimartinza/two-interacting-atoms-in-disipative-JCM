from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence_ali
import matplotlib as mpl
from entrelazamiento_lib import negativity_hor
import os 

# DEFINICIONES ----
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
g=0.01*w_0

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

def ket_to_bloch(v1,v2,ket):
    sz_1=pr(v1)-pr(v2)
    sx_1=v1*v2.dag()+v2*v1.dag()
    sy_1=-1j*v1*v2.dag()+1j*v2*v1.dag()
    
    return [expect(sx_1,ket),expect(sy_1,ket),expect(sz_1,ket)]

# JCM ----
## BLOCH CON DIRECCIONES ----
'''-------------------- GRAFICOS ESFERA BLOCH CON DIRECCIONES  ------------------'''

esfera=Bloch()
esfera.make_sphere()
delta=0
x=0
gamma=0.1*g
p=0.01*g
p0=0
p1=0.01*g
omega=np.sqrt(4*g**2+(delta-x)**2)

# # Simulacion numerica
num_ciclos=10
steps=3000*num_ciclos

T=2*np.pi/omega
t_final=num_ciclos*T
t=np.linspace(0,t_final,steps)
# delta_array=np.linspace(-10*g,10*g,201)
tita=np.arctan2(delta-x,2*g)
# h_vec=[[g]*len(delta_array),[0]*len(delta_array),delta_array/2-[x/2]*len(delta_array)]
h_vec=[g,0,delta/2-x/2]
h_vec=h_vec/np.sqrt(np.sum(h_vec_i**2 for h_vec_i in h_vec)) 
esfera.add_vectors(h_vec,colors='black')

phi=0
psi_perp=np.cos(tita/2)*e0+np.sin(tita/2)*g1
esfera.add_vectors(ket_to_bloch(e0,g1,psi_perp),colors='pink')
H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)

l_ops0=[np.sqrt(gamma)*a,np.sqrt(p0)*sm] #operadores de colapso/lindblad
l_ops1=[np.sqrt(gamma)*a,np.sqrt(p1)*sm] #operadores de colapso/lindblad

sol_u=mesolve(H,psi_perp,t)
sol_p0=mesolve(H,psi_perp,t,c_ops=l_ops0)
# sol_p1=mesolve(H,psi_perp,t,c_ops=l_ops1)
fg_p0,arg,eigenvals_t_d,psi_eig_p0 = fases(sol_p0)
# fg_p1,arg,eigenvals_t_d,psi_eig_p1 = fases(sol_p1)

ciclos_bloch=num_ciclos
points=ciclos_bloch*50

vBloch_tita=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
esfera.add_points(vBloch_tita,'s',colors='black')

vBloch_tita=vectorBloch(e0,g1,sol_p0.states,steps,ciclos_bloch,T,t_final,points)
esfera.add_points(vBloch_tita,'s',colors='lightblue')

vBloch_tita=vectorBloch(e0,g1,psi_eig_p0,steps,ciclos_bloch,T,t_final,points)
esfera.add_points(vBloch_tita,'s',colors='blue')

esfera.render()
esfera.show()
plt.show()

# TCM ----


## Leer negatividad ----