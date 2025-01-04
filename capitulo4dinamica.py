from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm
import matplotlib as mpl
import os


#DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
n=tensor(qeye(2),qeye(2),num(3))
sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,4])))
a=tensor(qeye(2),qeye(2),destroy(3))
sm1=tensor(sigmam(),qeye(2),qeye(3))
sp1=tensor(sigmap(),qeye(2),qeye(3))
sz1=tensor(sigmaz(),qeye(2),qeye(3))
sx1=tensor(sigmax(),qeye(2),qeye(3))
sm2=tensor(qeye(2),sigmam(),qeye(3))
sp2=tensor(qeye(2),sigmap(),qeye(3))
sz2=tensor(qeye(2),sigmaz(),qeye(3))
sx2=tensor(qeye(2),sigmax(),qeye(3))

#DEFINIMOS LOS VECTORES DE LA BASE
e=basis(2,0)
gr=basis(2,1)

e0=tensor(e,basis(3,0))
g0=tensor(gr,basis(3,0))
g1=tensor(gr,basis(3,1))
sx=tensor(sigmax(),qeye(3))
sy=tensor(sigmay(),qeye(3))
sz=tensor(sigmaz(),qeye(3))
sp=tensor(sigmap(),qeye(3))
sm=tensor(sigmam(),qeye(3))


ee0=tensor(e,e,basis(3,0)) #0
ee1=tensor(e,e,basis(3,1)) #1
ee2=tensor(e,e,basis(3,2)) #2

eg0=tensor(e,gr,basis(3,0)) #3
ge0=tensor(gr,e,basis(3,0)) #6

eg1=tensor(e,gr,basis(3,1)) #4
ge1=tensor(gr,e,basis(3,1)) #7

eg2=tensor(e,gr,basis(3,2)) #5
ge2=tensor(gr,e,basis(3,2)) #8

gg0=tensor(gr,gr,basis(3,0)) #9
gg1=tensor(gr,gr,basis(3,1)) #10
gg2=tensor(gr,gr,basis(3,2)) #11


SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

def beta_n(n_:int,k:float,J:float,x:float):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

'''#################################################################################################################################################
---------------------------------------------------REPRODUCCION RESULTADOS 1 ATOMO: ALPHA=0 -------------------------------------------------------------------
####################################################################################################################################################'''
alpha=0

w0=1
g=0.001*w0

p=0#0.005*g
gamma=0#.01*g

x=0
d=2*g

k=0
J=0


psi0=(eg0).unit()  #gg1#(tensor(tensor(e,gr)+tensor(gr,gr),basis(3,0)+basis(3,1))).unit()#1/10*(gg0*gg0.dag()+(eg0+ge0).unit()*(eg0+ge0).unit().dag()+(eg0-ge0).unit()*(eg0-ge0).unit().dag()+gg1*gg1.dag()+ee0*ee0.dag()+(eg1+ge1).unit()*(eg1+ge1).unit().dag()+(eg1-ge1).unit()*(eg1-ge1).unit().dag()+gg2*gg2.dag()+(eg2+ge2).unit()*(eg2+ge2).unit().dag()+(eg2-ge2).unit()*(eg2-ge2).unit().dag())
psi0Name='eg0'
# print(psi0)
steps=20000

T=2*np.pi/omega_general(1,1,d,g,k,J,x) 

t_final=T/np.sqrt(2)

acoplamiento='lineal'
def f():
    if acoplamiento=='lineal':
        return 1
    elif acoplamiento=='bs':
        return sqrtN

def pr(estado):
    return estado.unit()*estado.unit().dag()

'''##########---Hamiltoniano---##########'''

H=x*n2 + d/2*(sz1+alpha*sz2) + g*((sm1+alpha*sm2)*f()*a.dag()+(sp1+alpha*sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

'''#######---Simulacion numerica---#######'''
l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+alpha*sp2)] #OPERADORES DE COLAPSO

t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

sol_d=mesolve(H,psi0,t,c_ops=l_ops)


# points=200
# colors=mpl.colormaps['inferno'](np.linspace(0,1,points))
nrm = mpl.colors.Normalize(0, t_final)
colors = mpl.cm.inferno(nrm(t))
# print(colors)
colors_pob=mpl.colormaps['plasma'](np.linspace(0,1,4))

'''########---Estado total---###########'''

fg_d,arg,eigenvals_t_d = jcm.fases(sol_d)

# ops = [pr(gg0),pr(gg1),pr(eg0+ge0),pr(eg0-ge0),pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0),pr(eg2+ge2),pr(eg2-ge2),pr(ee1),
#             0.5*(sz1+sz2),sx1,sx2]

# ops_expect_d=np.empty((len(ops),len(sol_d.states)),dtype='complex')
# for i in range(len(sol_d.states)): 
#     for j in range(len(ops)):
#         ops_expect_d[j][i]=expect(ops[j],sol_d.states[i])

svn_tot=jcm.entropy_vn(eigenvals_t_d)
slin_tot=jcm.entropy_linear(sol_d.states)

'''#########---Atomo A-Cavidad----##########'''

atom_acavity_states_d=np.empty_like(sol_d.states)
vec_acavity=np.zeros((3,len(sol_d.states)))
for j in range(len(sol_d.states)):
    atom_acavity_states_d[j]=sol_d.states[j].ptrace([0,2])

sz1_02=pr(e0)-pr(g1)

sx1_02=e0*g1.dag()+g1*e0.dag()

sy1_02=-1j*e0*g1.dag()+1j*g1*e0.dag()

fig_pob_ac=plt.figure(figsize=(8,6))
ax=fig_pob_ac.add_subplot()
ax.plot(t/T,[atom_acavity_states_d[i][0][0] for i in range(steps)],label='e0',color=colors_pob[1])
ax.plot(t/T,[atom_acavity_states_d[i][4][4] for i in range(steps)],label='g1',color=colors_pob[3])
ax.plot(t/T,[atom_acavity_states_d[i][3][3] for i in range(steps)],label='g0',color=colors_pob[0])

ax.plot(t/T,[np.abs(atom_acavity_states_d[i][0][4]) for i in range(steps)],label='C',color=colors_pob[2],linestyle='dashed')


# ax.plot(t/T,[atom_acavity_states_d[i][2][2] for i in range(steps)],label='e2',color='black')
# ax.plot(t/T,[atom_acavity_states_d[i][1][1] for i in range(steps)],label='e1',color='black')
# ax.plot(t/T,[atom_acavity_states_d[i][5][5] for i in range(steps)],label='g2',color='black')

# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][0][1]) for i in range(steps)],label='C',color='black',linestyle='dashed')
# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][0][2]) for i in range(steps)],label='C',color='black',linestyle='dashed')
# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][0][3]) for i in range(steps)],label='C',color='black',linestyle='dashed')

# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][1][2]) for i in range(steps)],label='C',color='black',linestyle='dashed')
# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][1][3]) for i in range(steps)],label='C',color='black',linestyle='dashed')
# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][1][4]) for i in range(steps)],label='C',color='black',linestyle='dashed')

# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][2][3]) for i in range(steps)],label='C',color='black',linestyle='dashed')
# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][2][4]) for i in range(steps)],label='C',color='black',linestyle='dashed')

# ax.plot(t/T,[np.abs(atom_acavity_states_d[i][3][4]) for i in range(steps)],label='C',color='black',linestyle='dashed')



ax.set_xlim(0,t_final/T)
ax.set_ylim(0,1)
ax.set_xlabel('$t/T$',size=20)
ax.set_ylabel('Poblaciones',size=20)
ax.legend()
plt.show()


ops_acavity=[sx,sy,sz,sx1_02,sy1_02,sz1_02]

# expect_sx_acavity=[expect(atom_acavity_states_d[i],sx) for i in range(0,steps,int(steps/100))]
# expect_sy_acavity=[expect(atom_acavity_states_d[i],sy) for i in range(0,steps,int(steps/100))]
# expect_sz_acavity=[expect(atom_acavity_states_d[i],sz) for i in range(0,steps,int(steps/100))]

expect_sx1_02_acavity=[expect(atom_acavity_states_d[i],sx1_02) for i in range(steps)]
expect_sy1_02_acavity=[expect(atom_acavity_states_d[i],sy1_02) for i in range(steps)]
expect_sz1_02_acavity=[expect(atom_acavity_states_d[i],sz1_02) for i in range(steps)]

# vec_acavity=[expect_sx_acavity,expect_sy_acavity,expect_sz_acavity]
vec02_acavity=[expect_sx1_02_acavity,expect_sy1_02_acavity,expect_sz1_02_acavity]

# esfera1=Bloch()
# esfera1.make_sphere()
# esfera1.clear()
# esfera1.add_points(vec_acavity,'m',colors)
# esfera1.render()
# esfera1.save('eg0+ge0 bloch acavity 1.png')
# esfera1.show()

esfera02=Bloch()
esfera02.make_sphere()
esfera02.clear()
esfera02.add_points(vec02_acavity,'m')#,'m',colors)
esfera02.point_color=list(colors)

esfera02.render()
esfera02.save(f'./graficos/{psi0Name} bloch AC a={alpha} d={d/g} x={x/g} k={k/g} J={J/g} gamma={gamma/g} p={p/g}.png')

# esfera02.show()
# plt.show()
'''####---Atomo B---###'''

atom_b_states_d=np.empty_like(sol_d.states)
for j in range(len(sol_d.states)):
    atom_b_states_d[j]=sol_d.states[j].ptrace([1])

ops_b=[sigmax(),sigmay(),sigmaz()]
expect_sx_b=[expect(atom_b_states_d[i],sigmax()) for i in range(steps)]
expect_sy_b=[expect(atom_b_states_d[i],sigmay()) for i in range(steps)]
expect_sz_b=[expect(atom_b_states_d[i],sigmaz()) for i in range(steps)]

vec_b=[expect_sx_b,expect_sy_b,expect_sz_b]
esferaB=Bloch()
esferaB.set_label_convention('atomic')
esferaB.make_sphere()
esferaB.clear()
esferaB.add_points(vec_b,'m')#,colors_pob[0])#,'m',colors)
esferaB.point_color=list(colors)
esferaB.render()
esferaB.save(f'./graficos/{psi0Name} bloch B a={alpha} d={d/g} x={x/g} k={k/g} J={J/g} gamma={gamma/g} p={p/g}.png')

'''###---Atomo A-Atomo B---###'''

atoms_states_d=np.empty_like(sol_d.states)
for j in range(len(sol_d.states)):
    atoms_states_d[j]=sol_d.states[j].ptrace([0,1])  

svn_ab=jcm.entropy_vn_atom(atoms_states_d)
slin_ab=jcm.entropy_linear(atoms_states_d)
concu_ab=jcm.concurrence(atoms_states_d)

ops_atomos=[]




'''#################################################################################################################################################
---------------------------------------------------INSERTAR NOMBRE DE SECCION ACA-------------------------------------------------------------------
####################################################################################################################################################'''












'''#################################################################################################################################################
---------------------------------------------------INSERTAR NOMBRE DE SECCION ACA-------------------------------------------------------------------
####################################################################################################################################################'''



