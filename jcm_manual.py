import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import solve_ivp,cumulative_trapezoid
from jcm_lib import fases
import time

# print(tensor(basis(2,0),basis(2,0)))

#EN ESTE CODIGO VEMOS LA COMPARACION ENTRE EL METODO INTEGRAL Y LA FASE DE PANCHARATNAM. 
#ES MEJOR EL DE PANCHANATRAM PERO CREO QUE LAS ECS DIFS SON MAS RAPIDAS DE SIMULAR, ENTONCES QUIERO UTILIZAR
#EL METODO DE PANCHANATRAM PERO USANDO ECS DIFS COMO SOLVER.


def fases_anterior(delta, g, kappa, gamma, gamma_phi, t_int,rho_11,rho_12,rho_22):

  dotrho_12 = np.array(-1j*(delta) - kappa/2 - gamma/2 - gamma_phi)*np.array(rho_12) - 1j*g*(np.array(rho_22) - np.array(rho_11))
  epsilon = 0.5*np.array(np.array(rho_11) + np.array(rho_22) + np.sqrt((np.array(rho_11)-np.array(rho_22))**2 + 4*np.array(rho_12)*np.array(np.conj(rho_12))))

  # la función evaluada en el tiempo t = 0
  arg_int=np.imag(np.conj(dotrho_12)*np.array(rho_12))/((np.array(rho_22)-epsilon)**2+np.array(rho_12)*np.array(np.conj(rho_12)))
  phi_g = cumulative_trapezoid(np.imag(np.conj(dotrho_12)*np.array(rho_12))/((np.array(rho_22)-epsilon)**2+np.array(rho_12)*np.array(np.conj(rho_12))), t_int, initial=0)

  return phi_g, arg_int

w_0=1
g=0.05*w_0
delta=1e-6*g
x=0
omega=np.sqrt(4*g**2+(delta-x)**2)
T=2*np.pi/omega
t_final=10*T
# steps=100000
# t=np.linspace(0,t_final,steps)

steps_int=200000000
t_int=np.linspace(0,t_final,steps_int)
fg=np.zeros((3,steps_int))
gamma=0.1*g
p=0.01*g
gamma_z=0
fig_fg=plt.figure(figsize=(8,6))
ax_fg=fig_fg.add_subplot()
color=['black','red','blue']
label=['0','0.1g','0.2g']
for i,gamma,p in [[0,0,0],[1,0.01*g,0.1*g],[2,0.01*g,0.2*g]]:
  #Definimos ecs difs del problema a lo calculo numerico
  t_int_i=time.time()
  y0=np.array([0,1,0,0,0,0,0,0,0]) #rho00,rho11,im_rho12,re_rho12,rho22,rho33,im_rho34,re_rho34,rho44 con orden g0,e0,g1,e1,g2,.... 

  dot_rho=np.array([[0,p,0,0,gamma,0,0,0,0],
                    [0,-p,-2*g,0,0,gamma,0,0,0],
                    [0,g,-(p+gamma+2*gamma_z)/2,-(delta-x),-g,0,np.sqrt(2)*gamma,0,0],
                    [0,0,(delta-x),-(p+gamma+2*gamma_z)/2,0,0,0,np.sqrt(2)*gamma,0],
                    [0,0,2*g,0,-gamma,p,0,0,2*gamma],
                    [0,0,0,0,0,-p-gamma,-2*np.sqrt(2)*g,0,0],
                    [0,0,0,0,0,-np.sqrt(2)*g,-p/2-3*gamma/2-2*gamma_z,-(delta-3*x),np.sqrt(2)*g],
                    [0,0,0,0,0,0,(delta-3*x),-p/2-3*gamma/2-2*gamma_z,0],
                    [0,0,0,0,0,0,2*np.sqrt(2)*g,0,-2*gamma]])

  def f(t,y):
      return dot_rho@y

  sol=solve_ivp(f,(t_int[0],t_int[-1]),y0,t_eval=t_int,rtol=1e-3,atol=1e-6) #Default values are 1e-3 for rtol and 1e-6 for atol
  t_int_solve=time.time()
  print(t_int_solve-t_int_i,' tiempo de simulacion de ecs. difs.')
  # print(sol.t)
  # print(sol.y)
  rho00=sol.y[0]
  rho11=sol.y[1]
  im_rho12=sol.y[2]
  re_rho12=sol.y[3]
  rho22=sol.y[4]

  
  # fig_pob=plt.figure(figsize=(8,6))
  # ax_pob=fig_pob.add_subplot()
  # ax_pob.plot(sol.t/T,rho00,color='black',label='g0')
  # ax_pob.plot(sol.t/T,rho11,color='red',label='e0')
  # ax_pob.plot(sol.t/T,np.sqrt(im_rho12**2+re_rho12**2),color='magenta',linestyle='dashed',label='e0-g1')
  # ax_pob.plot(sol.t/T,rho22,color='blue',label='g1')
  # ax_pob.legend()



  fg_int,arg_int_fg=fases_anterior(delta,g,gamma,0,0,t_int,rho11,re_rho12+1j*im_rho12,rho22)
  fg_int_sub=fg_int[::int(steps_int/100000)]
  arg_int_fg=arg_int_fg[::int(steps_int/100000)]
  t_sub=sol.t[::int(steps_int/100000)]
  t_int_fg=time.time()
  print(-t_int_solve+t_int_fg,' tiempo de fg integral')
  ax_fg.plot(t_sub/T,fg_int_sub/np.pi,color=color[i],label=label[i]+' int')
  # ax_fg.plot(t_sub/T,arg_int_fg,color=color[i],linestyle='dashed',label='dx x '+label[i])


  t_pan_i=time.time()
  steps_qutip=100000
  t=np.linspace(0,t_final,steps_qutip)
  N_cav=2
  psi0=tensor(basis(2,0),basis(2,0))
  a_jcm=tensor(qeye(2),destroy(N_cav))
  n_jcm=tensor(qeye(2),num(N_cav))
  sm_jcm=tensor(sigmam(),qeye(N_cav))
  sz_jcm=tensor(sigmaz(),qeye(N_cav))
  H_jcm=delta/2*sz_jcm + x*n_jcm*n_jcm + g*(sm_jcm*a_jcm.dag()+sm_jcm.dag()*a_jcm)
  sol_qutip=mesolve(H_jcm,psi0,t,c_ops=[np.sqrt(gamma)*a_jcm,np.sqrt(p)*sm_jcm,np.sqrt(gamma_z)*sz_jcm])
  t_pan_solve=time.time()
  print(t_pan_solve-t_pan_i,' tiempo de simulacion de qutip')
  # print(H_jcm*psi0)

  fg_pan,argumento, eigenvalst,Psi=fases(sol_qutip)
  # fg_pan=fases_nuevo(sol_qutip,t)
  t_pan_fg=time.time()
  print(-t_pan_solve+t_pan_fg,' tiempo de fg panchanatram')


  # ax_fg.plot(t/T,nominador,color='red',linestyle='dashed',label='nom')
  ax_fg.plot(t/T,fg_pan/np.pi,color=color[i],linestyle='dashed',label=label[i]+' pan')
  # ax_fg.hlines(0,0,t[-1]/T)

ax_fg.vlines([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],0,10,'grey','dashdot')
ax_fg.legend()
plt.show()
