from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm
import matplotlib as mpl
import os
import time

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


SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.1)
plt.rc('figure.subplot',bottom=0.102)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)


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
    return 2*np.sqrt(-2*Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def energiasn1(j,g,d,x,k,J):
    if j==1: return (x-d)/2+k+np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==2: return (x-d)/2+k-np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==3: return -2*k-J
    else: 
        print('valor inesperado de j')
        exit()


'''#################################################################################################################################################
---------------------------------------------------ALPHA=1-------------------------------------------------------------------
####################################################################################################################################################'''

process_ti=time.process_time()
alpha=1

w0=1
g=0.001*w0

p=0.005*g
gamma=0.1*g

x=0#1*g
d=0#2*g#.5*g

k=0.5*g
J=0#.5*g


psi0=(gg1).unit()  #gg1#(tensor(tensor(e,gr)+tensor(gr,gr),basis(3,0)+basis(3,1))).unit()#1/10*(gg0*gg0.dag()+(eg0+ge0).unit()*(eg0+ge0).unit().dag()+(eg0-ge0).unit()*(eg0-ge0).unit().dag()+gg1*gg1.dag()+ee0*ee0.dag()+(eg1+ge1).unit()*(eg1+ge1).unit().dag()+(eg1-ge1).unit()*(eg1-ge1).unit().dag()+gg2*gg2.dag()+(eg2+ge2).unit()*(eg2+ge2).unit().dag()+(eg2-ge2).unit()*(eg2-ge2).unit().dag())
psi0Name='gg1'
prefijo=f'd={d/g} x={x/g} k={k/g} J={J/g}'

steps=40000

# T=2*np.pi/energiasn1(1,g,d,x,k,J)
# T=2*np.pi/(omega_general(1,1,d,g,k,J,x))
# T=2*np.pi/(-beta_n(1,k,J,x)/3+omega_general(1,1,d,g,k,J,x))
# print(T)
T=2*np.pi/omega_general(1,1,d,g,k,J,x)

# print(omega_general(1,2,d,g,k,J,x))
t_final=20*T
# ciclos_bloch=5
# points=2000

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
rho_ss=steadystate(H,l_ops)

# colors=mpl.colormaps['inferno'](np.linspace(0,1,steps))
def inferno(points:int):
    nrm = mpl.colors.Normalize(0, points)
    return mpl.cm.inferno(nrm(np.linspace(0,points,points)))
# colors=inferno(points)
colors_pob=mpl.colormaps['turbo'](np.linspace(0,1,4))
colors_coh=mpl.colormaps['turbo'](np.linspace(0,1,6))
colors_det=mpl.colormaps['copper'](np.linspace(0,1,5))
'''########---Estado total---###########'''

fg_d,arg,eigenvals_t_d = jcm.fases(sol_d)

process_t0=time.process_time()
print(f'0. SIMU Y CALCULO FG     ; {process_t0-process_ti}s ; -- ')

ops = [pr(ee0),pr(ee1),pr(ee2),pr(eg0),pr(eg1),pr(eg2),pr(ge0),pr(ge1),pr(ge2),pr(gg0),pr(gg1),pr(gg2)]
ops_name=['ee0','ee1','ee2','eg0','eg1','eg2','ge0','ge1','ge2','gg0','gg1','gg2']
ops_expect_d={}
for ops_i,ops_name_i in zip(ops,ops_name):
    ops_expect_d[ops_name_i]=[expect(sol_d.states[i],ops_i) for i in range(steps)]

# print(sol_d.states[2])
determinante_abc=[np.prod(eigenvals_t_d[i]) for i in range(steps)]

# ops_expect_d=np.empty((len(ops),len(sol_d.states)),dtype='complex')
# for i in range(len(sol_d.states)): 
#     for j in range(len(ops)):
#         ops_expect_d[j][i]=expect(ops[j],sol_d.states[i])

svn_tot=jcm.entropy_vn(eigenvals_t_d)
slin_tot=jcm.entropy_linear(sol_d.states)

fig_pob_ABC = plt.figure(figsize=(8,6))
ax_pob_ABC = fig_pob_ABC.add_subplot()
ax_pob_ABC.set_title('ABC')
ax_pob_ABC.set_xlabel('$t/T$')
ax_pob_ABC.set_ylabel('Amp. Prob. ')
ax_pob_ABC.set_ylim(0,1)
ax_pob_ABC.set_xlim(0,t_final/T)

# for i,ops_name_i in enumerate(ops_name):
#     ax_pob_ABC.plot(t/T,ops_expect_d[ops_name_i],color=inferno(12)[i])#,label=ops_name_i)

ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][9][9]) for i in range(steps)],color='black',label='gg0') #gg0

ax_pob_ABC.plot(t/T,[sol_d.states[i][3][3]/2+sol_d.states[i][6][6]/2+np.real(sol_d.states[i][3][6]) for i in range(steps)],color=(0, 255/255, 0),label='eg0+') #eg0+ge0
ax_pob_ABC.plot(t/T,[sol_d.states[i][3][3]/2+sol_d.states[i][6][6]/2-np.real(sol_d.states[i][3][6]) for i in range(steps)],color=(255/255, 0, 0),label='eg0-') #eg0-ge0
ax_pob_ABC.plot(t/T,[sol_d.states[i][10][10] for i in range(steps)],color=(0, 0, 255/255),label='gg1') #gg1

# ax_pob_ABC.hlines([rho_ss[9][9],rho_ss[3][3]/2+rho_ss[6][6]/2+np.real(rho_ss[3][6]),rho_ss[3][3]/2+rho_ss[6][6]/2-np.real(rho_ss[3][6]),rho_ss[10][10]],0,t_final/T,colors=['black',(0, 255/255, 0),(255/255, 0, 0),(0, 0, 255/255)],linestyles='dashed',alpha=0.5)

ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][3][9]+sol_d.states[i][6][9])/np.sqrt(2) for i in range(steps)],linestyle='dashed',color=(28/255, 176/255, 28/255)) #Coherenencia gg0/eg0+ge0
ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][3][9]-sol_d.states[i][6][9])/np.sqrt(2) for i in range(steps)],linestyle='dashed',color=(186/255, 23/255, 23/255)) #Coherenencia gg0/eg0-ge0

ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][9][10]) for i in range(steps)],linestyle='dashed',color=(0, 0, 170/255))

ax_pob_ABC.plot(t/T,[0.5*(sol_d.states[i][3][3]-2*np.imag(sol_d.states[i][3][6])-sol_d.states[i][6][6]) for i in range(steps)],linestyle='dashed',color=(255/255, 209/255, 0)) #Coherencia eg0+/eg0-

ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][3][10]+sol_d.states[i][3][10])/np.sqrt(2) for i in range(steps)],linestyle='dashed',color=(0, 209/255, 210/255)) #Coherenencia gg1/eg0-ge0
ax_pob_ABC.plot(t/T,[np.abs(sol_d.states[i][3][10]-sol_d.states[i][3][10])/np.sqrt(2) for i in range(steps)],linestyle='dashed',color=(235/255, 18/255, 235/255)) #Coherenencia gg1/eg0-ge0

# ax_pob_ABC.plot(t/T,np.array(ops_expect_d['eg0'])+np.array(ops_expect_d['ee0']),label='$e_A0_C$',color='red',marker=',',alpha=0.5) 
# ax_pob_ABC.plot(t/T,np.array(ops_expect_d['gg1'])+np.array(ops_expect_d['ge1']),label='$g_A1_C$',color='blue',marker=',',alpha=0.5)

ax_pob_ABC.legend(loc='upper right')
fig_pob_ABC.savefig(f'./graficos/{prefijo} {psi0Name} abc.png')


'''#########---Atomo A-Cavidad----##########'''

atom_acavity_states_d=np.empty_like(sol_d.states)
vec_acavity=np.zeros((3,len(sol_d.states)))
for j in range(len(sol_d.states)):
    atom_acavity_states_d[j]=sol_d.states[j].ptrace([0,2])

# determinante_ac=[np.linalg.det(atom_acavity_states_d[i]) for i in range(steps)]

# fg_ac_d,arg,eigenvals_t_d = jcm.fases(atom_acavity_states_d)
svn_ac=jcm.entropy_vn_atom(atom_acavity_states_d)

# slin_ac=jcm.entropy_linear(atom_acavity_states_d)

sz1_02=pr(e0)-pr(g1)

sx1_02=e0*g1.dag()+g1*e0.dag()

sy1_02=-1j*e0*g1.dag()+1j*g1*e0.dag()

fig_pob_AC=plt.figure(figsize=(8,6))
ax_AC=fig_pob_AC.add_subplot()

ax_AC.set_title('AC')
ax_AC.plot(t/T,[atom_acavity_states_d[i][0][0] for i in range(steps)],label='e0',color=colors_pob[1])
ax_AC.plot(t/T,[atom_acavity_states_d[i][4][4] for i in range(steps)],label='g1',color=colors_pob[3])
ax_AC.plot(t/T,[atom_acavity_states_d[i][3][3] for i in range(steps)],label='g0',color=colors_pob[0])

ax_AC.plot(t/T,[np.abs(atom_acavity_states_d[i][0][4]) for i in range(steps)],label='C',color=colors_pob[2],linestyle='dashed')


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



ax_AC.set_xlim(0,t_final/T)
ax_AC.set_ylim(0,1)
ax_AC.set_xlabel('$t/T$')
ax_AC.set_ylabel('Poblaciones')
ax_AC.legend()
fig_pob_AC.savefig(f'./graficos/{prefijo} {psi0Name} ac.png')




# ops_acavity=[sx,sy,sz,sx1_02,sy1_02,sz1_02]


# expect_sx1_02_acavity=[expect(atom_acavity_states_d[i],sx1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sy1_02_acavity=[expect(atom_acavity_states_d[i],sy1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sz1_02_acavity=[expect(atom_acavity_states_d[i],sz1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]


# esfera02=Bloch()
# esfera02.make_sphere()
# esfera02.clear()
# esfera02.set_point_marker(['o'])
# for i in range(points):
#     esfera02.add_points([expect_sx1_02_acavity[i],expect_sy1_02_acavity[i],expect_sz1_02_acavity[i]],colors=colors[i],)#,'m',colors)
    

# # esfera02.point_color=list(colors)

# esfera02.render()
# esfera02.save(f'./graficos/ch4 bloch/{psi0Name} bloch AC a={alpha} d={d/g} x={x/g} k={k/g} J={J/g} gamma={gamma/g} p={p/g}.png')



'''#########---Atomo B-Cavidad----##########'''

atom_bcavity_states_d=np.empty_like(sol_d.states)
vec_bcavity=np.zeros((3,len(sol_d.states)))
for j in range(len(sol_d.states)):
    atom_bcavity_states_d[j]=sol_d.states[j].ptrace([1,2])

# determinante_ac=[np.linalg.det(atom_acavity_states_d[i]) for i in range(steps)]

# fg_ac_d,arg,eigenvals_t_d = jcm.fases(atom_acavity_states_d)
svn_bc=jcm.entropy_vn_atom(atom_bcavity_states_d)

# slin_ac=jcm.entropy_linear(atom_acavity_states_d)

sz1_02=pr(e0)-pr(g1)

sx1_02=e0*g1.dag()+g1*e0.dag()

sy1_02=-1j*e0*g1.dag()+1j*g1*e0.dag()

fig_pob_BC=plt.figure(figsize=(8,6))
ax_BC=fig_pob_BC.add_subplot()

ax_BC.set_title('BC')
ax_BC.plot(t/T,[atom_bcavity_states_d[i][0][0] for i in range(steps)],label='e0',color=colors_pob[1])
ax_BC.plot(t/T,[atom_bcavity_states_d[i][4][4] for i in range(steps)],label='g1',color=colors_pob[3])
ax_BC.plot(t/T,[atom_bcavity_states_d[i][3][3] for i in range(steps)],label='g0',color=colors_pob[0])

ax_BC.plot(t/T,[np.abs(atom_bcavity_states_d[i][0][4]) for i in range(steps)],label='C',color=colors_pob[2],linestyle='dashed')


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



ax_BC.set_xlim(0,t_final/T)
ax_BC.set_ylim(0,1)
ax_BC.set_xlabel('$t/T$')
ax_BC.set_ylabel('Poblaciones')
ax_BC.legend()
fig_pob_BC.savefig(f'./graficos/{prefijo} {psi0Name} bc.png')




# ops_acavity=[sx,sy,sz,sx1_02,sy1_02,sz1_02]


# expect_sx1_02_acavity=[expect(atom_acavity_states_d[i],sx1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sy1_02_acavity=[expect(atom_acavity_states_d[i],sy1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sz1_02_acavity=[expect(atom_acavity_states_d[i],sz1_02) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]


# esfera02=Bloch()
# esfera02.make_sphere()
# esfera02.clear()
# esfera02.set_point_marker(['o'])
# for i in range(points):
#     esfera02.add_points([expect_sx1_02_acavity[i],expect_sy1_02_acavity[i],expect_sz1_02_acavity[i]],colors=colors[i],)#,'m',colors)
    

# # esfera02.point_color=list(colors)

# esfera02.render()
# esfera02.save(f'./graficos/ch4 bloch/{psi0Name} bloch AC a={alpha} d={d/g} x={x/g} k={k/g} J={J/g} gamma={gamma/g} p={p/g}.png')

process_t3=time.process_time()
# print(f'3. BLOCH AC           ; {process_t3-process_ti}s ; +{process_t3-process_t2}s')


'''####---Atomo B---###'''

# atom_b_states_d=np.empty_like(sol_d.states)
# for j in range(len(sol_d.states)):
#     atom_b_states_d[j]=sol_d.states[j].ptrace([1])

# # determinante_b=[np.linalg.det(atom_b_states_d[i]) for i in range(steps)]

# svn_b=jcm.entropy_vn_atom(atom_b_states_d)
# slin_b=jcm.entropy_linear(atom_b_states_d)


# fig_pob_B=plt.figure(figsize=(8,6))
# ax_B=fig_pob_B.add_subplot()
# ax_B.set_title('B')
# ax_B.set_xlim(0,t_final/T)

# ax_B.plot(t/T,[atom_b_states_d[i][0][0] for i in range(steps)],label='e',color=colors_pob[1])
# ax_B.plot(t/T,[atom_b_states_d[i][1][1] for i in range(steps)],label='g',color=colors_pob[3])

# ax_B.plot(t/T,[np.abs(atom_b_states_d[i][0][1]) for i in range(steps)],label='$C_{e,g}$',color=colors_pob[2],linestyle='dashed')


# ax_B.set_xlim(0,t_final/T)
# ax_B.set_ylim(0,1)
# ax_B.set_xlabel('$t/T$')
# ax_B.set_ylabel('Poblaciones')
# ax_B.legend()
# fig_pob_B.savefig(f'./graficos/{prefijo} {psi0Name} b.png')



# ops_b=[sigmax(),sigmay(),sigmaz()]
# expect_sx_b=[expect(atom_b_states_d[i],sigmax()) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sy_b=[expect(atom_b_states_d[i],sigmay()) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
# expect_sz_b=[expect(atom_b_states_d[i],sigmaz()) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]

# # vec_b=[expect_sx_b,expect_sy_b,expect_sz_b]
# esferaB=Bloch()
# esferaB.set_label_convention('atomic')
# esferaB.make_sphere()
# esferaB.clear()
# esferaB.set_point_marker(['o'])
# for i in range(points):
#     esferaB.add_points([expect_sx_b[i],expect_sy_b[i],expect_sz_b[i]],colors=colors[i],)#,'m',colors)
    

# esferaB.render()
# esferaB.save(f'./graficos/ch4 bloch/{psi0Name} bloch B a={alpha} d={d/g} x={x/g} k={k/g} J={J/g} gamma={gamma/g} p={p/g}.png')



'''###---Atomo A-Atomo B---###'''

atoms_states_d=np.empty_like(sol_d.states)
for j in range(len(sol_d.states)):
    atoms_states_d[j]=sol_d.states[j].ptrace([0,1])  

# determinante_ab=[np.linalg.det(atoms_states_d[i]) for i in range(steps)]

fig_pob_AB=plt.figure(figsize=(8,6))
ax_AB=fig_pob_AB.add_subplot()
ax_AB.set_title('AB')
ax_AB.set_xlim(0,t_final/T)

ax_AB.plot(t/T,[atoms_states_d[i][0][0] for i in range(steps)],label='ee',color=colors_pob[3])
ax_AB.plot(t/T,[atoms_states_d[i][1][1] for i in range(steps)],label='eg',color=colors_pob[2])
ax_AB.plot(t/T,[atoms_states_d[i][2][2] for i in range(steps)],label='ge',color=colors_pob[1])
ax_AB.plot(t/T,[atoms_states_d[i][3][3] for i in range(steps)],label='gg',color=colors_pob[0])



# ax_AB.plot(t/T,[np.abs(atoms_states_d[i][0][1]) for i in range(steps)],label='$C_{ee,eg}$',color='black',linestyle='dashed')
# ax_AB.plot(t/T,[np.abs(atoms_states_d[i][0][2]) for i in range(steps)],label='$C_{ee,ge}$',color='black',linestyle='dashed')
# ax_AB.plot(t/T,[np.abs(atoms_states_d[i][0][3]) for i in range(steps)],label='$C_{ee,gg}$',color='black',linestyle='dashed')

ax_AB.plot(t/T,[np.abs(atoms_states_d[i][1][2]) for i in range(steps)],label='$C_{eg,ge}$',color='black',linestyle='dashed')
ax_AB.plot(t/T,[np.abs(atoms_states_d[i][1][3]) for i in range(steps)],label='$C_{eg,gg}$',color='red',linestyle='dashed')


# ax_AB.plot(t/T,[np.abs(atoms_states_d[i][2][3]) for i in range(steps)],label='$C_{ge,gg}$',color='black',linestyle='dashed')


ax_AB.set_xlim(0,t_final/T)
ax_AB.set_ylim(0,1)
ax_AB.set_xlabel('$t/T$')
ax_AB.set_ylabel('Poblaciones')
ax_AB.legend()
fig_pob_AB.savefig(f'./graficos/{prefijo} {psi0Name} ab.png')




svn_ab=jcm.entropy_vn_atom(atoms_states_d)
slin_ab=jcm.entropy_linear(atoms_states_d)
concu_ab=jcm.concurrence(atoms_states_d)

fig_concu=plt.figure(figsize=(8,6))
ax_concu=fig_concu.add_subplot()
ax_concu.set_title('Concu')
ax_concu.set_ylabel('Concu')
ax_concu.set_xlabel('$t/T$')
ax_concu.plot(t/T,concu_ab,color='black')
ax_concu.set_ylim(0,1)
ax_concu.set_xlim(0,t_final/T)
fig_concu.savefig(f'./graficos/{prefijo} {psi0Name} concu.png')



'''##############---Cavidad---############'''

cavity_states_d=np.empty_like(sol_d.states)
for j in range(len(sol_d.states)):
    cavity_states_d[j]=sol_d.states[j].ptrace([2])

# determinante_c=[np.linalg.det(cavity_states_d[i]) for i in range(steps)]

svn_c=jcm.entropy_vn_atom(cavity_states_d)
slin_c=jcm.entropy_linear(cavity_states_d)


fig_entropia=plt.figure(figsize=(8,6))
# ax_slin=fig_entropia.add_subplot(121)
# ax_slin.set_title('$S_{lin}$')
# ax_slin.set_xlim(0,t_final/T)

# ax_slin.plot(t/T,slin_tot,color='black',label='ABC',zorder=1,linewidth=2,marker='o',markevery=int(steps/10*T/t_final))

# ax_slin.plot(t/T,slin_ab,color='red',label='AB')
# ax_slin.plot(t/T,slin_c,color='red',linestyle='dashed',label='C',lw=2)

# ax_slin.plot(t/T,slin_ac,color='blue',label='AC')
# ax_slin.plot(t/T,slin_b,color='blue',linestyle='dashed',label='B')

# # ax_slin.plot(t/T,slin_tot-slin_ab-slin_b,color='green',marker='o',label='$S_T(AB)$')
# ax_slin.legend()
# ax_slin.hlines(0.5,0,t_final/T,colors='grey',linestyles='dashed')


ax_svn=fig_entropia.add_subplot()
# ax_svn.set_title('$S_{vN}$')
ax_svn.set_xlim(0,t_final/T)

ax_svn.plot(t/T,svn_tot,color='black',label='ABC',zorder=1,linewidth=2,marker='o',markevery=int(steps/10*T/t_final))

ax_svn.plot(t/T,svn_ab,color='red',label='AB')

ax_svn.plot(t/T,svn_ac,color='blue',label='AC')
ax_svn.plot(t/T,svn_bc,color='blue',linestyle='dashed',label='BC',lw=2)
ax_svn.plot(t/T,svn_c,color='red',linestyle='dashed',label='C',lw=2)

ax_svn.plot(t/T,svn_ac+svn_bc-svn_c,color='green',label='SSA')

# ax_svn.plot(t/T,svn_b,color='blue',linestyle='dashed',label='B')
# ax_svn.plot(t/T,svn_tot-svn_ab-svn_b,color='green',marker='o',label='$S_T(AB)$')
ax_svn.hlines(np.log(2),0,t_final/T,colors='grey',linestyles='dashed')
ax_svn.set_xlabel('$t/T$')
ax_svn.set_ylabel('$S_{vN}$')
ax_svn.legend()
fig_entropia.savefig(f'./graficos/{prefijo} {psi0Name} svn.png')


fig_det=plt.figure(figsize=(8,6))
ax_det=fig_det.add_subplot()
ax_det.set_xlabel('$t/T$')
ax_det.set_ylabel('$det(\\rho)$')
ax_det.plot(t/T,determinante_abc,color=colors_det[0],label='ABC')


plt.show()

