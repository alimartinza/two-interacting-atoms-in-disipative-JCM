from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence_ali
from entrelazamiento_lib import negativity_hor

import matplotlib as mpl
from matplotlib import cm
import os 

# Definiciones ----


## 1 Estados QUTIP ---- 

#DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
N_c=3
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

#DEFINIMOS LOS VECgORES DE LA BASE
e=basis(2,0)
gr=basis(2,1)

# e0=tensor(e,basis(N_c,0))
# g0=tensor(gr,basis(N_c,0))
# g1=tensor(gr,basis(N_c,1))
# sx=tensor(sigmax(),qeye(N_c))
# sy=tensor(sigmay(),qeye(N_c))
# sz=tensor(sigmaz(),qeye(N_c))
# sp=tensor(sigmap(),qeye(N_c))
# sm=tensor(sigmam(),qeye(N_c))


ee0=tensor(e,e,basis(N_c,0)) #0
ee1=tensor(e,e,basis(N_c,1)) #1
ee2=tensor(e,e,basis(N_c,2)) #2

eg0=tensor(e,gr,basis(N_c,0)) #3
ge0=tensor(gr,e,basis(N_c,0)) #6

eg1=tensor(e,gr,basis(N_c,1)) #4
ge1=tensor(gr,e,basis(N_c,1)) #7

eg2=tensor(e,gr,basis(N_c,2)) #5
ge2=tensor(gr,e,basis(N_c,2)) #8

gg0=tensor(gr,gr,basis(N_c,0)) #9
gg1=tensor(gr,gr,basis(N_c,1)) #10
gg2=tensor(gr,gr,basis(N_c,2)) #11


##2 Config matplotlib ----
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.18)
plt.rc('figure.subplot',bottom=0.12)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)

## 3 Funciones ----
def beta_n(n_:int,k,J:float,x):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d,g:float,k,J:float,x,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d,g:float,k,J:float,x,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

def Q_n(n_:int,d,g:float,k,J:float,x):
    return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

def R_n(n_:int,d,g:float,k,J:float,x):
    return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

def theta_n(n_:int,d,g:float,k,J:float,x):
    return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

def En_(n_:int,j:int,d,g:float,k,J:float,x):
    return -beta_n(n_,k,J,x)/3+2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def Omega_n_ij(n_,i,j,d,g,k,J,x):
    return En_(n_,j,d,g,k,J,x)-En_(n_,i,d,g,k,J,x)

def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def rabi_freq(n_:int,j1:int,j2:int,d:float,g:float,k:float,J:float,x:float):
    return omega_general(n_,j2,d,g,k,J,x)-omega_general(n_,j1,d,g,k,J,x)

def energiasn1(j,g,d,x,k,J):
    if j==1: return (x-d)/2+k+np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==2: return (x-d)/2+k-np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==3: return -2*k-J
    else: 
        print('valor inesperado de j')
        exit()

def H_tcm(g,d,x,k,J):
    return x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
os.chdir(script_path)

def pr(estado):
    return estado.unit()*estado.unit().dag()


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

def l_ops(gamma,p):
    return [np.sqrt(gamma)*a,np.sqrt(p)*sm1,np.sqrt(p)*sm2]


## Evolucion exacta ----
w_0=1
g=0.01*w_0
delta=0*g
x=0
k=0
J=0

T=2*np.pi/rabi_freq(2,1,2,delta,g,k,J,x)

def h11(n):
    return x*(n-2)**2+delta+J
def h22(n):
    return x*(n-1)**2-J+2*k
def h33(n):
    return x*n**2-delta+J
def h12(n):
    return np.sqrt(2*(n-1))*g
def h23(n):
    return np.sqrt(2*n)*g

def H_array(n):
    return np.array([[h11(n),h12(n),0],[h12(n),h22(n),h23(n)],[0,h23(n),h33(n)]])


E_j_num,u_j_num=H_tcm(g,delta,x,k,J).eigenstates()
n0=2

def c_ij(i_,j_):
    N=np.sqrt( (h12(n0)*h23(n0))**2 + (h23(n0)*(En_(n0,i_,delta,g,k,J,x)-h11(n0)))**2 +((En_(n0,i_,delta,g,k,J,x)-h22(n0))*(En_(n0,i_,delta,g,k,J,x)-h11(n0))-h12(n0)**2)**2 )
    if j_==1:
        return h12(n0)*h23(n0)/N
    elif j_==2:
        return h23(n0)*(En_(n0,i_,delta,g,k,J,x)-h11(n0))/N
    elif j_==3:
        return ((En_(n0,i_,delta,g,k,J,x)-h22(n0))*(En_(n0,i_,delta,g,k,J,x)-h11(n0))-h12(n0)**2)/N


def u_i(i):
    return np.array([c_ij(i,1),c_ij(i,2),c_ij(i,3)])

def phi_i(i):
    return c_ij(1,i)*u_i(1)+c_ij(2,i)*u_i(2)+c_ij(3,i)*u_i(3)

# phi_j=[np.cos(alpha)*np.sin(beta),np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1),np.sin(alpha)*np.exp(1j*phi2)]

# a=sum_l(sum_j phi_j (sum_k c_kj c_kl e^-i E_k t))|phi_k>

def psi_t(t,alpha,beta,phi1,phi2):
    E1=En_(n0,1,delta,g,k,J,x)
    E2=En_(n0,2,delta,g,k,J,x)
    E3=En_(n0,3,delta,g,k,J,x)
    return [np.cos(alpha)*np.sin(beta)*(c_ij(1,1)*c_ij(1,1)*np.exp(-1j*E1*t)+c_ij(2,1)*c_ij(2,1)*np.exp(-1j*E2*t)+c_ij(3,1)*c_ij(3,1)*np.exp(-1j*E3*t))+np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1)*(c_ij(1,2)*c_ij(1,1)*np.exp(-1j*E1*t)+c_ij(2,2)*c_ij(2,1)*np.exp(-1j*E2*t)+c_ij(3,2)*c_ij(3,1)*np.exp(-1j*E3*t))+np.sin(alpha)*np.exp(1j*phi2)*(c_ij(1,3)*c_ij(1,1)*np.exp(-1j*E1*t)+c_ij(2,3)*c_ij(2,1)*np.exp(-1j*E2*t)+c_ij(3,3)*c_ij(3,1)*np.exp(-1j*E3*t)),
            np.cos(alpha)*np.sin(beta)*(c_ij(1,1)*c_ij(1,2)*np.exp(-1j*E1*t)+c_ij(2,1)*c_ij(2,2)*np.exp(-1j*E2*t)+c_ij(3,1)*c_ij(3,2)*np.exp(-1j*E3*t))+np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1)*(c_ij(1,2)*c_ij(1,2)*np.exp(-1j*E1*t)+c_ij(2,2)*c_ij(2,2)*np.exp(-1j*E2*t)+c_ij(3,2)*c_ij(3,2)*np.exp(-1j*E3*t))+np.sin(alpha)*np.exp(1j*phi2)*(c_ij(1,3)*c_ij(1,2)*np.exp(-1j*E1*t)+c_ij(2,3)*c_ij(2,2)*np.exp(-1j*E2*t)+c_ij(3,3)*c_ij(3,2)*np.exp(-1j*E3*t)),
            np.cos(alpha)*np.sin(beta)*(c_ij(1,1)*c_ij(1,3)*np.exp(-1j*E1*t)+c_ij(2,1)*c_ij(2,3)*np.exp(-1j*E2*t)+c_ij(3,1)*c_ij(3,3)*np.exp(-1j*E3*t))+np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1)*(c_ij(1,2)*c_ij(1,3)*np.exp(-1j*E1*t)+c_ij(2,2)*c_ij(2,3)*np.exp(-1j*E2*t)+c_ij(3,2)*c_ij(3,3)*np.exp(-1j*E3*t))+np.sin(alpha)*np.exp(1j*phi2)*(c_ij(1,3)*c_ij(1,3)*np.exp(-1j*E1*t)+c_ij(2,3)*c_ij(2,3)*np.exp(-1j*E2*t)+c_ij(3,3)*c_ij(3,3)*np.exp(-1j*E3*t))]

alpha=np.pi/4
beta=np.pi/4
phi1=np.pi/2
phi2=0

t=np.linspace(0,10*T,10000)
psi_exacto=psi_t(t,alpha,beta,phi1,phi2)


pob_ee0_exacto=np.power(np.abs(psi_exacto[0]),2)
pob_eg1_exacto=np.power(np.abs(psi_exacto[1]),2)
pob_gg2_exacto=np.power(np.abs(psi_exacto[2]),2)

### Sanity check ----

# psi0_qt=np.cos(alpha)*np.sin(beta)*ee0+np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1)*(eg1+ge1).unit()+np.sin(alpha)*np.exp(1j*phi2)*gg0
# sol=mesolve(H_tcm(g,delta,x,k,J),psi0_qt,t,e_ops=[pr(ee0),pr((eg1+ge1).unit()),pr(gg2)])

# fig=plt.figure(figsize=(8,6))
# ax=fig.add_subplot()
# ax.set_title('Check evolucion exacta')
# ax.plot(t/T,pob_ee0_exacto,color='red')
# ax.plot(t/T,pob_eg1_exacto,color='blue')
# ax.plot(t/T,pob_gg2_exacto,color='black')

# ax.plot(t/T,sol.expect[0],color='red',linestyle='dashed',marker='o',markevery=100)
# ax.plot(t/T,sol.expect[1],color='blue',linestyle='dashed',marker='o',markevery=100)
# ax.plot(t/T,sol.expect[2],color='black',linestyle='dashed',marker='o',markevery=100)

# ax.set_xlabel('t/T')
# ax.set_ylabel('Pob')
# plt.grid()
# plt.show()

#Krylov State Complexity ----

## Lanczos algorithm ----
psi0=np.array([np.cos(alpha)*np.sin(beta),np.cos(alpha)*np.cos(beta)*np.exp(1j*phi1),np.sin(alpha)*np.exp(1j*phi2)])

# print(np.sum(np.conjugate(psi0)*psi0))

psi1=H_array(n0) @ psi0
a0=np.sum(np.conjugate(psi0)*psi1)

A1=psi1-a0*psi0

b1=np.sqrt(np.sum(np.conjugate(A1)*A1))

if b1==0:
    K1=[0,0,0]
    K2=[0,0,0]
    print('a1=0',f'|A1>={A1}',f'|K2>={K2}')

else:
    print(f'b1={b1}')
    K1=A1/b1
    print(f'|K1|^2={np.sum(np.conjugate(K1)*K1)}')

    # a1=np.sum(np.conjugate(K1)*(H_array(n0) @ K1))
    a1=np.vdot(K1,H_array(n0) @ K1)
    A2=H_array(n0) @ K1-a1*K1-b1*psi0
    b2=np.sqrt(np.sum(np.conjugate(A2)*A2))

    if np.abs(b2)<1e-12:
        print('b2 dio 0')
        print('b2=0',f'|A2>={A2}')
        K2=np.array([0,0,0])
    else:
        K2=A2/b2

##Chequeo de ortogonalidad ----
print('----- Orthogonality check -----')
print("<psi0|K1> =", np.vdot(psi0, K1))
print("<psi0|K2> =", np.vdot(psi0, K2))
print("<K1|K2>   =", np.vdot(K1, K2))


## 
print('psi_exacto?')
print(len(psi_exacto[0]))

## Resultados Lanczos ----
print('------ Resultados de Lanczos --------',f'|psi0>={psi0}',f'|K1>={K1}',f'|K2>={K2}')

#Ahora vamos a ver las coordenadas de |psi(t)> en la base {|Kn>}
c0=np.ones(len(psi_exacto[0]))
c1=np.ones(len(psi_exacto[0]))
c2=np.ones(len(psi_exacto[0]))
psi_K=np.ones((len(psi_exacto[0]),3))
print('len(psiecaxto)',len(psi_exacto))
for j in range(len(psi_exacto[0])):
    c0[j]=np.vdot(psi0,np.array([psi_exacto[0][j],psi_exacto[1][j],psi_exacto[2][j]]))
    c1[j]=np.vdot(K1,np.array([psi_exacto[0][j],psi_exacto[1][j],psi_exacto[2][j]]))
    c2[j]=np.vdot(K2,np.array([psi_exacto[0][j],psi_exacto[1][j],psi_exacto[2][j]]))
    psi_K[j]=c0[j]*psi0+c1[j]*K1+c2[j]*K2

print(f'len(c0)={len(c0)}',f'len(psi_K)={len(psi_K)}',f'len(psi_exacto[0])={len(psi_exacto[0])}')

#Intentemos de ver si funciona... tecnicamente seria |psi(t)>= c0(t)*psi0+c1(t)*K1+c2(t)*K2
pob_ee0_K=np.zeros(len(psi_K))
pob_eg1_K=np.zeros(len(psi_K))
pob_gg2_K=np.zeros(len(psi_K))
for j in range(len(psi_K)):
    pob_ee0_K[j]=np.power(np.abs(psi_K[j][0]),2)
    pob_eg1_K[j]=np.power(np.abs(psi_K[j][1]),2)
    pob_gg2_K[j]=np.power(np.abs(psi_K[j][2]),2)

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot()
ax.set_title('Check evolucion en base K')
ax.plot(t/T,np.abs(c0),color='red')
ax.plot(t/T,np.abs(c1),color='blue')
ax.plot(t/T,np.abs(c2),color='black')

ax.plot(t/T,pob_ee0_K+pob_eg1_K+pob_gg2_K,color='black',linestyle='dashed',marker='o',markevery=10)

# ax.plot(t/T,pob_ee0_K,color='red',linestyle='dashed',marker='o',markevery=10)
# ax.plot(t/T,pob_eg1_K,color='blue',linestyle='dashed',marker='o',markevery=10)
# ax.plot(t/T,pob_gg2_K,color='black',linestyle='dashed',marker='o',markevery=10)

ax.set_xlabel('t/T')
ax.set_ylabel('Pob')
plt.grid()
plt.show()




# print('---- En_ analitico -----')
# print(En_(n0,1,delta,g,k,J,x))
# print(En_(n0,2,delta,g,k,J,x))
# print(En_(n0,3,delta,g,k,J,x))

# print('---- En numerico ----')
# for j in [1,6,11]:
#     print(E_j[j])

# print
# print('---- u1 ------')
# print(u1.dag()*u_j[1])
# print(u1.dag()*u_j[6])
# print(u1.dag()*u_j[11])

# print('---- u2 ------')
# print(u2.dag()*u_j[1])
# print(u2.dag()*u_j[6])
# print(u2.dag()*u_j[11])

# print('---- u3 ------')
# print(u3.dag()*u_j[1])
# print(u3.dag()*u_j[6])
# print(u3.dag()*u_j[11])
