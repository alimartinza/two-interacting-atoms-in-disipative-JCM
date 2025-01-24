from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm
import matplotlib as mpl
import os
import time
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

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
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def rabi_freq(n_:int,j1:int,j2:int,d:float,g:float,k:float,J:float,x:float):
    return omega_general(n_,j2,d,g,k,J,x)-omega_general(n_,j1,d,g,k,J,x)

w_0=1
g=0.01*w_0

gamma=0.1*g
p=0.005*g

d=0
x=0

J=0
k=0.5*g

T=2*np.pi/np.abs(rabi_freq(2,2,3,d,g,k,J,x))
t_final=3*T
steps=10000

psi0=(eg1+ge1).unit()
psi0Name='eg1+ge1'

param_list=np.linspace(0,5*g,20)

param_name='$\Delta$'
save_name='k=0.5'
folder_name='delta/eg1+ge1'


cmap = mpl.colormaps['inferno'](np.linspace(param_list[0]/g,param_list[-1]/g,len(param_list)))   # Viridis colormap with as many colors as CSV files

data_dis=np.zeros((len(param_list),steps))

t=np.linspace(0,t_final,steps)


acoplamiento='lineal'
def f():
    if acoplamiento=='lineal':
        return 1
    elif acoplamiento=='bs':
        return sqrtN

def pr(estado):
    return estado.unit()*estado.unit().dag()

colors_pob2=mpl.colormaps['inferno'](np.linspace(0,1,10))

pob_gg0=np.zeros((len(param_list),steps),dtype='complex')
pob_gg1=np.zeros((len(param_list),steps),dtype='complex')
pob_gg2=np.zeros((len(param_list),steps),dtype='complex')
pob_eg0sim=np.zeros((len(param_list),steps),dtype='complex')
pob_eg0asim=np.zeros((len(param_list),steps),dtype='complex')
pob_eg1sim=np.zeros((len(param_list),steps),dtype='complex')
pob_eg1asim=np.zeros((len(param_list),steps),dtype='complex')
pob_ee0=np.zeros((len(param_list),steps),dtype='complex')

coh_gg1_eg0sim=np.zeros((len(param_list),steps),dtype='complex')
coh_gg1_eg0asim=np.zeros((len(param_list),steps),dtype='complex')
coh_eg0sim_eg0asim=np.zeros((len(param_list),steps),dtype='complex')

coh_gg2_eg1sim=np.zeros((len(param_list),steps),dtype='complex')
coh_eg1sim_ee0=np.zeros((len(param_list),steps),dtype='complex')
coh_gg2_ee0=np.zeros((len(param_list),steps),dtype='complex')
coh_eg1sim_eg1asim=np.zeros((len(param_list),steps),dtype='complex')
coh_gg2_eg1asim=np.zeros((len(param_list),steps),dtype='complex')
coh_ee0_eg1asim=np.zeros((len(param_list),steps),dtype='complex')


for l,d in enumerate(param_list):
    jcm.simu_disip(w_0,g,k,J,d,x,gamma,p,1,psi0,t_final,steps)

    '''##########---Hamiltoniano---##########'''
    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    '''#######---Simulacion numerica---#######'''
    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)] #OPERADORES DE COLAPSO

    sol_d=mesolve(H,psi0,t,c_ops=l_ops)
    pob_gg0[l]=[(sol_d.states[i][9][9]) for i in range(steps)]

    pob_gg1[l]=[sol_d.states[i][10][10] for i in range(steps)]
    pob_eg0sim[l]=[sol_d.states[i][3][3]/2+sol_d.states[i][6][6]/2+np.real(sol_d.states[i][3][6]) for i in range(steps)]
    pob_eg0asim[l]=[sol_d.states[i][3][3]/2+sol_d.states[i][6][6]/2-np.real(sol_d.states[i][3][6]) for i in range(steps)]

    pob_gg2[l]=[sol_d.states[i][11][11] for i in range(steps)]
    pob_eg1sim[l]=[(sol_d.states[i][4][4]+sol_d.states[i][7][7]+sol_d.states[i][4][7]+sol_d.states[i][7][4])/2 for i in range(steps)]
    pob_ee0[l]=[sol_d.states[i][0][0] for i in range(steps)]
    pob_eg1asim[l]=[(sol_d.states[i][4][4]+sol_d.states[i][7][7]-sol_d.states[i][4][7]-sol_d.states[i][7][4])/2 for i in range(steps)]

    coh_gg1_eg0sim[l]=[(sol_d.states[i][3][10]+sol_d.states[i][3][10])/np.sqrt(2) for i in range(steps)]
    coh_gg1_eg0asim[l]=[(sol_d.states[i][3][10]-sol_d.states[i][3][10])/np.sqrt(2) for i in range(steps)]
    coh_eg0sim_eg0asim[l]=[0.5*(sol_d.states[i][3][3]-2*np.imag(sol_d.states[i][3][6])-sol_d.states[i][6][6]) for i in range(steps)]
    
    coh_gg2_eg1sim[l]=np.array([(sol_d.states[i][11][4]+sol_d.states[i][11][7])/np.sqrt(2) for i in range(steps)])
    coh_gg2_ee0[l]=np.array([sol_d.states[i][11][0] for i in range(steps)])
    coh_gg2_eg1asim[l]=np.array([(sol_d.states[i][11][4]-sol_d.states[i][11][7])/np.sqrt(2) for i in range(steps)])
    coh_eg1sim_ee0[l]=np.array([(sol_d.states[i][4][0]+sol_d.states[i][7][0])/np.sqrt(2) for i in range(steps)])
    coh_eg1sim_eg1asim[l]=np.array([(sol_d.states[i][4][4]-sol_d.states[i][7][7]-sol_d.states[i][4][7]+sol_d.states[i][7][4])/2 for i in range(steps)])
    coh_ee0_eg1asim[l]=np.array([(sol_d.states[i][4][0]-sol_d.states[i][7][0])/np.sqrt(2) for i in range(steps)])

fig_pob_1 = plt.figure(figsize=(8,6))
ax_pob_1 = fig_pob_1.add_subplot()
ax_pob_1.set_title('N=1')
ax_pob_1.set_xlabel('$t/T$')
ax_pob_1.set_ylabel('Amp. Prob. ')
ax_pob_1.set_ylim(0,1)
ax_pob_1.set_xlim(0,t_final/T)

# for i,ops_name_i in enumerate(ops_name):
#     ax_pob_1.plot(t/T,ops_expect_d[ops_name_i],color=inferno(12)[i])#,label=ops_name_i)

line_pob_gg0,=ax_pob_1.plot(t/T,pob_gg0[0],color=colors_pob2[0],label='gg0') #gg0

line_pob_eg0sim,=ax_pob_1.plot(t/T,pob_eg0sim[0],color=colors_pob2[1],label='eg0+') #eg0+ge0
line_pob_eg0asim,=ax_pob_1.plot(t/T,pob_eg0asim[0],color=colors_pob2[2],label='eg0-') #eg0-ge0
line_pob_gg1,=ax_pob_1.plot(t/T,pob_gg1[0],color=colors_pob2[3],label='gg1') #gg1

# ax_pob_1.hlines([rho_ss[9][9],rho_ss[3][3]/2+rho_ss[6][6]/2+np.real(rho_ss[3][6]),rho_ss[3][3]/2+rho_ss[6][6]/2-np.real(rho_ss[3][6]),rho_ss[10][10]],0,t_final/T,colors=['black',(0, 255/255, 0),(255/255, 0, 0),(0, 0, 255/255)],linestyles='dashed',alpha=0.5)

line_coh_eg0sim_eg0asim_re,=ax_pob_1.plot(t/T,coh_eg0sim_eg0asim[0],linestyle='dashed',color=colors_pob2[7],marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherencia eg0+/eg0-
line_coh_eg0sim_eg0asim_im,=ax_pob_1.plot(t/T,coh_eg0sim_eg0asim[0],linestyle='dashed',color=colors_pob2[7],label='eg0+;eg0-',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherencia eg0+/eg0-

line_coh_gg1_eg0sim_re,=ax_pob_1.plot(t/T,coh_gg1_eg0sim[0],linestyle='dashed',color=colors_pob2[8],marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherenencia gg1/eg0-ge0
line_coh_gg1_eg0sim_im,=ax_pob_1.plot(t/T,coh_gg1_eg0sim[0],linestyle='dashed',color=colors_pob2[8],label='gg1;eg0+',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherenencia gg1/eg0-ge0

line_coh_gg1_eg0asim_re,=ax_pob_1.plot(t/T,coh_gg1_eg0asim[0],linestyle='dashed',color=colors_pob2[9],marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherenencia gg1/eg0-ge0
line_coh_gg1_eg0asim_im,=ax_pob_1.plot(t/T,coh_gg1_eg0asim[0],linestyle='dashed',color=colors_pob2[9],label='gg1;eg1-',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #Coherenencia gg1/eg0-ge0

# ax_pob_1.plot(t/T,np.array(ops_expect_d['eg0'])+np.array(ops_expect_d['ee0']),label='$e_A0_C$',color='red',marker=',',alpha=0.5) 
# ax_pob_1.plot(t/T,np.array(ops_expect_d['gg1'])+np.array(ops_expect_d['ge1']),label='$g_A1_C$',color='blue',marker=',',alpha=0.5)

ax_pob_1.legend(loc='upper right')
# fig_pob_1.savefig(f'./graficos/{prefijo} {psi0Name} abc.png')

marker_size=3
marker='o'
imag_threshold=0.03
fig_pob_2=plt.figure(figsize=(8,6))
ax_pob_2=fig_pob_2.add_subplot()

ops_name=['gg2','C(gg2;eg1+)','eg1+','C(ee0;gg2)','ee0','C(gg2;eg1-)','C(ee0;eg1+)','C(eg1+;eg1-)','C(ee0;eg1-)','eg1-']
ax_pob_2.set_title('no abs')
line_pob_gg2,=ax_pob_2.plot(t/T,pob_gg2[0],color=colors_pob2[0],label='gg2') #gg2



line_coh_gg2_eg1sim_re,=ax_pob_2.plot(t/T,np.real(coh_gg2_eg1sim[0]),color=colors_pob2[1],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #
line_coh_gg2_eg1sim_im,=ax_pob_2.plot(t/T,np.imag(coh_gg2_eg1sim[0]),color=colors_pob2[1],linestyle='dashed',label='gg2;eg1+',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #
# index_imag_gg2_eg1sim=[i for i in range(len(coh_gg2_eg1sim)) if np.abs(np.imag(coh_gg2_eg1sim[i]))>imag_threshold]
# ax3.scatter(1/T*t[index_imag_gg2_eg1sim],coh_gg2_eg1sim[index_imag_gg2_eg1sim],s=marker_size,marker=marker,color=colors_pob2[1])


line_pob_eg1sim,=ax_pob_2.plot(t/T,pob_eg1sim[0],color=colors_pob2[2],label='eg1+') #


line_coh_gg2_ee0_re,=ax_pob_2.plot(t/T,np.real(coh_gg2_ee0[0]),color=colors_pob2[3],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #gg2;ee0
line_coh_gg2_ee0_im,=ax_pob_2.plot(t/T,np.imag(coh_gg2_ee0[0]),color=colors_pob2[3],linestyle='dashed',label='gg2;ee0',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #gg2;ee0
# index_imag_gg2_ee0=[i for i in range(len(coh_gg2_ee0)) if np.abs(np.imag(coh_gg2_ee0[i]))>imag_threshold]
# ax3.scatter(1/T*t[index_imag_gg2_ee0],coh_gg2_ee0[index_imag_gg2_ee0],s=marker_size,marker=marker,color=colors_pob2[3])


line_pob_ee0,=ax_pob_2.plot(t/T,pob_ee0[0],color=colors_pob2[4],label='ee0') #ee0

line_coh_gg2_eg1asim_re,=ax_pob_2.plot(t/T,np.real(coh_gg2_eg1asim[0]),color=colors_pob2[5],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #
line_coh_gg2_eg1asim_im,=ax_pob_2.plot(t/T,np.imag(coh_gg2_eg1asim[0]),color=colors_pob2[5],linestyle='dashed',label='gg2;eg1-',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #
# index_imag_gg2_eg1asim=[i for i in range(len(coh_gg2_eg1asim)) if np.abs(np.imag(coh_gg2_eg1asim[i]))>imag_threshold]
# ax3.scatter(1/T*t[index_imag_gg2_eg1asim],coh_gg2_eg1asim[index_imag_gg2_eg1asim],s=marker_size,marker=marker,color=colors_pob2[5])

line_coh_eg1sim_ee0_re,=ax_pob_2.plot(t/T,np.real(coh_eg1sim_ee0[0]),color=colors_pob2[6],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #
line_coh_eg1sim_ee0_im,=ax_pob_2.plot(t/T,np.imag(coh_eg1sim_ee0[0]),color=colors_pob2[6],linestyle='dashed',label='eg1+;ee0',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #
# index_imag_eg1sim_ee0=[i for i in range(len(coh_eg1sim_ee0)) if np.abs(np.imag(coh_eg1sim_ee0[i]))>imag_threshold]
# ax3.scatter(1/T*t[index_imag_eg1sim_ee0],coh_eg1sim_ee0[index_imag_eg1sim_ee0],s=marker_size,marker=marker,color=colors_pob2[6])

line_coh_eg1sim_eg1asim_re,=ax_pob_2.plot(t/T,np.real(coh_eg1sim_eg1asim[0]),color=colors_pob2[7],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #
line_coh_eg1sim_eg1asim_im,=ax_pob_2.plot(t/T,np.imag(coh_eg1sim_eg1asim[0]),color=colors_pob2[7],linestyle='dashed',label='eg1+;eg1-',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #
# index_imag_eg1sim_eg1asim=[i for i in range(len(coh_eg1sim_eg1asim)) if np.imag(coh_eg1sim_eg1asim[i])>imag_threshold]
# ax3.scatter(1/T*t[index_imag_eg1sim_eg1asim],coh_eg1sim_eg1asim[index_imag_eg1sim_eg1asim],s=marker_size,marker=marker,color=colors_pob2[7])

line_coh_ee0_eg1asim_re,=ax_pob_2.plot(t/T,np.real(coh_ee0_eg1asim[0]),color=colors_pob2[8],linestyle='dashed',marker='s',markevery=int(steps/50*T/t_final),alpha=0.5) #
line_coh_ee0_eg1asim_im,=ax_pob_2.plot(t/T,np.imag(coh_ee0_eg1asim[0]),color=colors_pob2[8],linestyle='dashed',label='ee0;eg1-',marker='o',markevery=int(steps/50*T/t_final),alpha=0.5) #
# index_imag_ee0_eg1asim=[i for i in range(len(coh_ee0_eg1asim)) if np.imag(coh_ee0_eg1asim[i])>imag_threshold]
# ax3.scatter(1/T*t[index_imag_ee0_eg1asim],coh_ee0_eg1asim[index_imag_ee0_eg1asim],s=marker_size,marker=marker,color=colors_pob2[8])

line_pob_eg1asim,=ax_pob_2.plot(t/T,pob_eg1asim[0],color=colors_pob2[9],label='eg1-') #
ax_pob_2.legend()



cmap = mpl.colormaps['viridis']   # Viridis colormap with as many colors as CSV files
norm = mpl.colors.Normalize(vmin=param_list[0]/g, vmax=param_list[-1]/g)

# Add the colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # ScalarMappable requires an array, but we don't need it
cbar1 = plt.colorbar(sm, ax=ax_pob_1, orientation='vertical')
cbar2 = plt.colorbar(sm, ax=ax_pob_2, orientation='vertical')
cbar1.set_label(param_name+'$/g$')
cbar2.set_label(param_name+'$/g$')

# Create a black rectangle to indicate the current position in the colorbar
rect_height = 1 / len(param_list)  # Height of the rectangle
current_color_rect1 = Rectangle((0, 0), width=1, height=rect_height, color='black', lw=2, transform=cbar1.ax.transAxes)
current_color_rect2 = Rectangle((0, 0), width=1, height=rect_height, color='black', lw=2, transform=cbar2.ax.transAxes)
cbar1.set_label(param_name+'$/g$')
cbar2.set_label(param_name+'$/g$')
cbar1.ax.add_patch(current_color_rect1)  # Add the rectangle to the colorbar axes
cbar2.ax.add_patch(current_color_rect2)  # Add the rectangle to the colorbar axes

# Define the initialization function
def init1():
    """Initialize the plot with empty data."""
    # line_u.set_data([], [])
    line_pob_gg0.set_data([],[])

    line_pob_gg1.set_data([],[])
    line_pob_eg0sim.set_data([],[])
    line_pob_eg0asim.set_data([],[])

    line_coh_gg1_eg0sim_re.set_data([],[])
    line_coh_gg1_eg0sim_im.set_data([],[])
    line_coh_gg1_eg0asim_re.set_data([],[])
    line_coh_gg1_eg0asim_im.set_data([],[])
    line_coh_eg0sim_eg0asim_re.set_data([],[])
    line_coh_eg0sim_eg0asim_im.set_data([],[])
    return line_pob_gg0,line_pob_gg1,line_pob_eg0sim,line_pob_eg0asim,line_coh_gg1_eg0sim_re,line_coh_gg1_eg0sim_im,line_coh_gg1_eg0asim_re,line_coh_gg1_eg0asim_im,line_coh_eg0sim_eg0asim_re,line_coh_eg0sim_eg0asim_im,

def init2():    
    line_pob_gg2.set_data([],[])
    line_pob_eg1sim.set_data([],[])
    line_pob_ee0.set_data([],[])
    line_pob_eg1asim.set_data([],[])

    line_coh_gg2_eg1sim_re.set_data([],[])
    line_coh_gg2_eg1sim_im.set_data([],[])
    line_coh_gg2_ee0_re.set_data([],[])
    line_coh_gg2_ee0_im.set_data([],[])
    line_coh_gg2_eg1asim_re.set_data([],[])
    line_coh_gg2_eg1asim_im.set_data([],[])
    line_coh_eg1sim_ee0_re.set_data([],[])
    line_coh_eg1sim_ee0_im.set_data([],[])
    line_coh_eg1sim_eg1asim_re.set_data([],[])
    line_coh_eg1sim_eg1asim_im.set_data([],[])
    line_coh_ee0_eg1asim_re.set_data([],[])
    line_coh_ee0_eg1asim_im.set_data([],[])
    return line_pob_gg2,line_pob_eg1sim,line_pob_ee0,line_pob_eg1asim,line_coh_gg2_eg1sim_re,line_coh_gg2_eg1sim_im,line_coh_gg2_ee0_re,line_coh_gg2_ee0_im,line_coh_gg2_eg1asim_re,line_coh_gg2_eg1asim_im,line_coh_eg1sim_ee0_re,line_coh_eg1sim_ee0_im,line_coh_eg1sim_eg1asim_re,line_coh_eg1sim_eg1asim_im,line_coh_ee0_eg1asim_re,line_coh_ee0_eg1asim_im,current_color_rect1
    # Define the update function for each frame
def update1(frame):
    """Read the CSV data and update the plot."""
    # Update the plot data
    # line_u.set_data(g*t, data_uni[frame])
    # line_d.set_data(t/T, data_dis[frame])
    line_pob_gg0.set_data(t/T,pob_gg0[frame])

    line_pob_gg1.set_data(t/T,pob_gg1[frame])
    line_pob_eg0sim.set_data(t/T,pob_eg0sim[frame])
    line_pob_eg0asim.set_data(t/T,pob_eg0asim[frame])

    line_coh_gg1_eg0sim_re.set_data(t/T,np.real(coh_gg1_eg0sim[frame]))
    line_coh_gg1_eg0sim_im.set_data(t/T,np.imag(coh_gg1_eg0sim[frame]))
    line_coh_gg1_eg0asim_re.set_data(t/T,np.real(coh_gg1_eg0asim[frame]))
    line_coh_gg1_eg0asim_im.set_data(t/T,np.imag(coh_gg1_eg0asim[frame]))
    line_coh_eg0sim_eg0asim_re.set_data(t/T,np.real(coh_eg0sim_eg0asim[frame]))
    line_coh_eg0sim_eg0asim_im.set_data(t/T,np.imag(coh_eg0sim_eg0asim[frame]))
    current_color_rect1.set_y(frame / len(param_list))  # Adjust y based on current frame
    return line_pob_gg0,line_pob_gg1,line_pob_eg0sim,line_pob_eg0asim,line_coh_gg1_eg0sim_re,line_coh_gg1_eg0sim_im,line_coh_gg1_eg0asim_re,line_coh_gg1_eg0asim_im,line_coh_eg0sim_eg0asim_re,line_coh_eg0sim_eg0asim_im
def update2(frame):    
    line_pob_gg2.set_data(t/T,pob_gg2[frame])
    line_pob_eg1sim.set_data(t/T,pob_eg1sim[frame])
    line_pob_ee0.set_data(t/T,pob_ee0[frame])
    line_pob_eg1asim.set_data(t/T,pob_eg1asim[frame])

    line_coh_gg2_eg1sim_re.set_data(t/T,np.real(coh_gg2_eg1sim[frame]))
    line_coh_gg2_eg1sim_im.set_data(t/T,np.imag(coh_gg2_eg1sim[frame]))
    line_coh_gg2_ee0_re.set_data(t/T,np.real(coh_gg2_ee0[frame]))
    line_coh_gg2_ee0_im.set_data(t/T,np.imag(coh_gg2_ee0[frame]))
    line_coh_gg2_eg1asim_re.set_data(t/T,np.real(coh_gg2_eg1asim[frame]))
    line_coh_gg2_eg1asim_im.set_data(t/T,np.imag(coh_gg2_eg1asim[frame]))
    line_coh_eg1sim_ee0_re.set_data(t/T,np.real(coh_eg1sim_ee0[frame]))
    line_coh_eg1sim_ee0_im.set_data(t/T,np.imag(coh_eg1sim_ee0[frame]))
    line_coh_eg1sim_eg1asim_re.set_data(t/T,np.real(coh_eg1sim_eg1asim[frame]))
    line_coh_eg1sim_eg1asim_im.set_data(t/T,np.imag(coh_eg1sim_eg1asim[frame]))
    line_coh_ee0_eg1asim_re.set_data(t/T,np.real(coh_ee0_eg1asim[frame]))
    line_coh_ee0_eg1asim_im.set_data(t/T,np.imag(coh_ee0_eg1asim[frame]))

    # Move the rectangle to the current position in the colorbar (keep it black)
    current_color_rect2.set_y(frame / len(param_list))  # Adjust y based on current frame
    return line_pob_gg2,line_pob_eg1sim,line_pob_ee0,line_pob_eg1asim,line_coh_gg2_eg1sim_re,line_coh_gg2_eg1sim_im,line_coh_gg2_ee0_re,line_coh_gg2_ee0_im,line_coh_gg2_eg1asim_re,line_coh_gg2_eg1asim_im,line_coh_eg1sim_ee0_re,line_coh_eg1sim_ee0_im,line_coh_eg1sim_eg1asim_re,line_coh_eg1sim_eg1asim_im,line_coh_ee0_eg1asim_re,line_coh_ee0_eg1asim_im,current_color_rect2

for l in range(len(param_list)):
    update1(l)
    update2(l)
    fig_pob_1.savefig(f'./graficos/ch4 anim/{folder_name}/n=1 {save_name} {l}.png')
    fig_pob_2.savefig(f'./graficos/ch4 anim/{folder_name}/n=2 {save_name} {l}.png')

# Show the plot
plt.legend()
plt.show()
