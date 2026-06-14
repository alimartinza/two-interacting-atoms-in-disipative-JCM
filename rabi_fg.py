import jcm_lib as jcm
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd



# Rabi en JCM ----

N_c=50

# M=np.zeros((4*N_c,4*N_c))
# M[0,3*N_c]=1
# M[1,3*N_c+1]=1
# M[2,N_c]=1/np.sqrt(2)
# M[2,2*N_c]=1/np.sqrt(2)
# M[3,N_c]=1/np.sqrt(2)
# M[3,2*N_c]=-1/np.sqrt(2)

# for ii in range(1,N_c-1):
#     M[4*ii,3*N_c+1+ii]=1
# for ii in range(1,N_c-1):
#     M[4*ii+1,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
# for ii in range(1,N_c-1):
#     M[4*ii+2,ii-1]=1

e=basis([2],[0])
gr=basis([2],[1])

def en(n):
    return tensor(e,basis(N_c,n))

def gn(n):
    return tensor(gr,basis(N_c,n))



n=tensor(qeye(2),num(N_c))
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
a=tensor(qeye(2),destroy(N_c))
sm=tensor(sigmam(),qeye(N_c))
sp=tensor(sigmap(),qeye(N_c))
sz=tensor(sigmaz(),qeye(N_c))
sx=tensor(sigmax(),qeye(N_c))


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


"""Returns: 
data_tcm: dataframe de pandas con los datos de la simulacion unitaria
data_rabi: dataframe de pandas con los datos de la simulacion disipativa
-Las keys de los dfs son:
,t,pr(gg0),pr(gg1),pr(eg0+ge0),pr(eg0-ge0),pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0),pr(eg2+ge2),pr(eg2-ge2),pr(ee1),1/2 <sz1+sz2>,<sx1>,<sx2>,0;1,0;2,0;3,0;4,0;5,0;6,0;7,0;8,0;9,0;10,0;11,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,2;3,2;4,2;5,2;6,2;7,2;8,2;9,2;10,2;11,3;4,3;5,3;6,3;7,3;8,3;9,3;10,3;11,4;5,4;6,4;7,4;8,4;9,4;10,4;11,5;6,5;7,5;8,5;9,5;10,5;11,6;7,6;8,6;9,6;10,6;11,7;8,7;9,7;10,7;11,8;9,8;10,8;11,9;10,9;11,10;11,FG,S von Neuman tot,S lineal tot,S vN atom,S lin atom,Concu atom,Eigenvalue 0,Eigenvalue 1,Eigenvalue 2,Eigenvalue 3,Eigenvalue 4,Eigenvalue 5,Eigenvalue 6,Eigenvalue 7,Eigenvalue 8,Eigenvalue 9,Eigenvalue 10,Eigenvalue 11
    """
#DEFINIMOS CUAL MODELO VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO

w_r=2*np.pi*6*1e9 #Hz
g=0.001*w_r
delta=0.0001*g
w_q=w_r+delta 

gamma=0.1*g
p=0.05*gamma

x=0 #2*np.pi*1e6 #Hz

omega=np.sqrt(4*g**2+(delta-x)**2)
T=2*np.pi/omega
ciclos=30
t_final=ciclos*T

steps=3000*ciclos
psi0=en(0)
'''---Hamiltoniano---'''

H_jcm=w_r*n+x*n2 + w_q/2*sz + g*(sm*a.dag()+sp*a) 
H_rabi=w_r*n+x*n2 + w_q/2*sz + g*sx*(a+a.dag()) 
'''---Simulacion numerica---'''
t_0=time.time()
l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp]

t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION

sol_jcm_u=mesolve(H_jcm,psi0,t,c_ops=[])
sol_jcm_d=mesolve(H_jcm,psi0,t,c_ops=l_ops)
t_1=time.time()
print(t_1-t_0,'s en simular jcm')

sol_rabi_u=mesolve(H_rabi,psi0,t,c_ops=[])
sol_rabi_d=mesolve(H_rabi,psi0,t,c_ops=l_ops)
t_2=time.time()
print(t_2-t_1,'s en simular rabi')

fg_jcm_u,arg = jcm.fases(sol_jcm_u,False)
fg_jcm_d,arg,eigenvals_t_tcm,eigenvecs_t_jcm = jcm.fases(sol_jcm_d,True,2)

# fg_viejo_jcm_u,arg,_,_ = jcm.fases_viejo(sol_jcm_u)
# fg_viejo_jcm_d,arg,eigenvals_t_jcm_viejo,eigenvecs_t_jcm_viejo = jcm.fases_viejo(sol_jcm_d)

fg_rabi_u,arg = jcm.fases(sol_rabi_u,False)
fg_rabi_d,arg,eigenvals_t_rabi,eigenvecs_t_rabi = jcm.fases(sol_rabi_d,True,2)

# fg_viejo_rabi_u,arg,_,_ = jcm.fases_viejo(sol_rabi_u)
# fg_viejo_rabi_d,arg,eigenvals_t_rabi_viejo,eigenvecs_t_rabi_viejo = jcm.fases_viejo(sol_rabi_d)

t_3=time.time()
print(t_3-t_2,'s en computar las FGs')


# data_tcm=pd.DataFrame()
# data_rabi=pd.DataFrame()
# data_tcm['t']=t
# data_rabi['t']=t
# #Hacemos un array de las coherencias y las completamos con el for
# coherencias_tcm={'0;1':[],'0;2':[],'0;3':[],'0;4':[],'0;5':[],'0;6':[],'0;7':[],'0;8':[],'0;9':[],'0;10':[],'0;11':[],
#                         '1;2':[],'1;3':[],'1;4':[],'1;5':[],'1;6':[],'1;7':[],'1;8':[],'1;9':[],'1;10':[],'1;11':[],
#                                 '2;3':[],'2;4':[],'2;5':[],'2;6':[],'2;7':[],'2;8':[],'2;9':[],'2;10':[],'2;11':[],
#                                         '3;4':[],'3;5':[],'3;6':[],'3;7':[],'3;8':[],'3;9':[],'3;10':[],'3;11':[],
#                                                 '4;5':[],'4;6':[],'4;7':[],'4;8':[],'4;9':[],'4;10':[],'4;11':[],
#                                                         '5;6':[],'5;7':[],'5;8':[],'5;9':[],'5;10':[],'5;11':[],
#                                                                 '6;7':[],'6;8':[],'6;9':[],'6;10':[],'6;11':[],
#                                                                         '7;8':[],'7;9':[],'7;10':[],'7;11':[],
#                                                                                 '8;9':[],'8;10':[],'8;11':[],
#                                                                                         '9;10':[],'9;11':[],
#                                                                                                 '10;11':[]}

# coherencias_rabi=coherencias_tcm

#DEFINIMOS LOS OPERADORES A LOS QUE QUEREMOS QUE EL SOLVER TOME VALOR MEDIO. LOS PROYECTORES NOS DAN LAS POBLACIONES
ops_nomb=['pr(gg0)','pr(gg1)','pr(eg0)','pr(ge0)','pr(gg2)','pr(eg1)','pr(ge1)','pr(ee0)','pr(eg2)','pr(ge2)',
    'pr(ee1)','pr(ee2)','1/2 <Sz>','<sx1>','<sx2>','<n>'] #NOMBRES PARA EL LEGEND DEL PLOT
ops = [pr(gn(0)),pr(en(0)),pr(gn(1)),pr(en(1)),pr(gn(2)),pr(en(2)),pr(gn(3)),pr(en(3)),pr(gn(4)),pr(en(4)),pr(gn(5)),pr(en(5)),pr(gn(6)),pr(en(6)),pr(gn(7)),pr(en(7)),pr(gn(8)),pr(en(8)),pr(gn(9)),n]

expectStartTime=time.process_time()
ops_expect_jcm_u=np.empty((len(ops),len(sol_jcm_u.states)))
for i in range(len(sol_jcm_u.states)): 
    for j in range(len(ops)):
        ops_expect_jcm_u[j][i]=expect(ops[j],sol_jcm_u.states[i])

ops_expect_jcm_d=np.empty((len(ops),len(sol_jcm_d.states)))
for i in range(len(sol_jcm_d.states)): 
    for j in range(len(ops)):
        ops_expect_jcm_d[j][i]=expect(ops[j],sol_jcm_d.states[i])

ops_expect_rabi_u=np.empty((len(ops),len(sol_rabi_u.states)))
for i in range(len(sol_rabi_u.states)): 
    for j in range(len(ops)):
        ops_expect_rabi_u[j][i]=expect(ops[j],sol_rabi_u.states[i])



ops_expect_rabi_d=np.empty((len(ops),len(sol_rabi_d.states)))
for i in range(len(sol_rabi_d.states)): 
    for j in range(len(ops)):
        ops_expect_rabi_d[j][i]=expect(ops[j],sol_rabi_d.states[i])


fig_pob=plt.figure(figsize=(12,9),tight_layout=True)
ax1=fig_pob.add_subplot(221)
ax1.plot(t/T,ops_expect_jcm_u[0],color='black',label='P|g0>')
ax1.plot(t/T,ops_expect_jcm_u[1],color='red',label='P|e0>')
ax1.plot(t/T,ops_expect_jcm_u[2],color='blue',label='P|g1>')

ax1.plot(t/T,ops_expect_rabi_u[0],color='black',linestyle='dashed')
ax1.plot(t/T,ops_expect_rabi_u[1],color='red',linestyle='dashed')
ax1.plot(t/T,ops_expect_rabi_u[2],color='blue',linestyle='dashed')
ax1.set_xlabel('t/T')
ax1.set_ylabel(r'$P_{|ggn\rangle}$')
ax1.legend()

ax2=fig_pob.add_subplot(222)
ax2.plot(t/T,ops_expect_jcm_u[3],color='black',label='P|e1>')
ax2.plot(t/T,ops_expect_jcm_u[4],color='red',label='P|g2>')
ax2.plot(t/T,ops_expect_jcm_u[5],color='blue',label='P|e2>')
ax2.plot(t/T,ops_expect_jcm_u[6],color='green',label='P|g3>')

ax2.plot(t/T,ops_expect_rabi_u[3],color='black',linestyle='dashed')
ax2.plot(t/T,ops_expect_rabi_u[4],color='red',linestyle='dashed')
ax2.plot(t/T,ops_expect_rabi_u[5],color='blue',linestyle='dashed')
ax2.plot(t/T,ops_expect_rabi_u[6],color='green',linestyle='dashed')
ax2.set_xlabel('t/T')
ax2.legend()

ax3=fig_pob.add_subplot(223)
ax3.plot(t/T,ops_expect_jcm_u[3],color='black',label='P|e3>')
ax3.plot(t/T,ops_expect_jcm_u[4],color='red',label='P|g4>')
ax3.plot(t/T,ops_expect_jcm_u[5],color='blue',label='P|e4>')
ax3.plot(t/T,ops_expect_jcm_u[6],color='green',label='P|g5>')

ax3.plot(t/T,ops_expect_rabi_u[3],color='black',linestyle='dashed')
ax3.plot(t/T,ops_expect_rabi_u[4],color='red',linestyle='dashed')
ax3.plot(t/T,ops_expect_rabi_u[5],color='blue',linestyle='dashed')
ax3.plot(t/T,ops_expect_rabi_u[6],color='green',linestyle='dashed')
ax3.set_xlabel('t/T')
ax3.legend()

ax4=fig_pob.add_subplot(224)
ax4.plot(t/T,ops_expect_jcm_u[-1],color='black',label='<n>_u jcm')
ax4.plot(t/T,ops_expect_jcm_d[-1],color='black',linestyle='dashed')
ax4.plot(t/T,ops_expect_rabi_u[-1],color='red',label='<n>_u rabi')
ax4.plot(t/T,ops_expect_rabi_d[-1],color='red',linestyle='dashed')
ax4.set_ylabel(r'$\langle \hat{n} \rangle$')
ax4.set_xlabel('t/T')
ax4.legend()

fig_fg=plt.figure(figsize=(8,6))
ax=fig_fg.add_subplot()
ax.set_title('FG')
ax.plot(t/T,fg_jcm_u/np.pi,color='black',label='JCM')
ax.plot(t/T,fg_jcm_d/np.pi,color='black',linestyle='dashed')

# ax.plot(t/T,fg_viejo_jcm_u/np.pi,color='blue')
# ax.plot(t/T,fg_viejo_jcm_d/np.pi,color='blue',linestyle='dashed')

ax.plot(t/T,fg_rabi_u/np.pi,color='red',label='RABI')
ax.plot(t/T,fg_rabi_d/np.pi,color='red',linestyle='dashed')

# ax.plot(t/T,fg_viejo_rabi_u/np.pi,color='orange')
# ax.plot(t/T,fg_viejo_rabi_d/np.pi,color='orange',linestyle='dashed')

ax.set_xlabel('t/T')
ax.set_ylabel(r'$\phi_G/\pi$')
ax.legend()


esfera_jcm=Bloch()
esfera_jcm.make_sphere()
esfera_jcm.fig.suptitle("JCM", fontsize=16)

ciclos_bloch=ciclos
points=ciclos_bloch*50

vBloch_tita=vectorBloch(en(0),gn(1),sol_jcm_u.states,steps,ciclos_bloch,T,t_final,points)
esfera_jcm.add_points(vBloch_tita,'s',colors='black')

vBloch_tita=vectorBloch(en(0),gn(1),sol_jcm_d.states,steps,ciclos_bloch,T,t_final,points)
esfera_jcm.add_points(vBloch_tita,'s',colors='lightblue')

vBloch_tita=vectorBloch(en(0),gn(1),eigenvecs_t_jcm[0],steps,ciclos_bloch,T,t_final,points)
esfera_jcm.add_points(vBloch_tita,'s',colors='blue')

esfera_jcm.render()
esfera_jcm.show()


esfera_rabi=Bloch()
esfera_rabi.make_sphere()
esfera_rabi.fig.suptitle("RABI", fontsize=16)
ciclos_bloch=5
points=ciclos_bloch*50

vBloch_tita=vectorBloch(en(0),gn(1),sol_rabi_u.states,steps,ciclos_bloch,T,t_final,points)
esfera_rabi.add_points(vBloch_tita,'s',colors='black')

vBloch_tita=vectorBloch(en(0),gn(1),sol_rabi_d.states,steps,ciclos_bloch,T,t_final,points)
esfera_rabi.add_points(vBloch_tita,'s',colors='lightblue')

vBloch_tita=vectorBloch(en(0),gn(1),eigenvecs_t_rabi[0],steps,ciclos_bloch,T,t_final,points)
esfera_rabi.add_points(vBloch_tita,'s',colors='blue')

esfera_rabi.render()
esfera_rabi.show()


fig_pobs_evecs=plt.figure(figsize=(8,6))
ax_pobs_evecs=fig_pobs_evecs.add_subplot()
ax_pobs_evecs.set_title('Pob eigenvecs')
ax_pobs_evecs.plot(t/T,expect(pr(gn(1)),eigenvecs_t_jcm[0]),color='red',label='g1')
ax_pobs_evecs.plot(t/T,expect(pr(en(0)),eigenvecs_t_jcm[0]),color='blue',label='e0')

ax_pobs_evecs.plot(t/T,expect(pr(gn(0)),eigenvecs_t_rabi[0]),color='black',linestyle='dashed',label='g0')

ax_pobs_evecs.plot(t/T,expect(pr(gn(1)),eigenvecs_t_rabi[0]),color='red',linestyle='dashed',label='g1')
ax_pobs_evecs.plot(t/T,expect(pr(en(0)),eigenvecs_t_rabi[0]),color='blue',linestyle='dashed',label='e0')

ax_pobs_evecs.plot(t/T,expect(pr(gn(2)),eigenvecs_t_rabi[0]),color='yellow',linestyle='dashed',label='g2')
ax_pobs_evecs.plot(t/T,expect(pr(en(1)),eigenvecs_t_rabi[0]),color='green',linestyle='dashed',label='e1')

ax_pobs_evecs.plot(t/T,expect(pr(gn(3)),eigenvecs_t_rabi[0]),color='orange',linestyle='dashed',label='g3')
ax_pobs_evecs.plot(t/T,expect(pr(en(2)),eigenvecs_t_rabi[0]),color='lightblue',linestyle='dashed',label='e2')

ax_pobs_evecs.set_xlabel('t/T')
ax_pobs_evecs.set_ylabel('Pob')
ax_pobs_evecs.legend()
plt.show()