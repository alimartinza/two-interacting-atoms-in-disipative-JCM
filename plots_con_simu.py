from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from jcm_lib import simu_unit_y_disip,anim_univsdis


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

# from mpl_toolkits.mplot3d import axes3d


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

psi0=(ee0+gg2).unit()
psi0Name="ee0+gg2"
steps=5000
t_final=500000
w_0=1
J=0
g=0.001*w_0
p=0.005*g
d=0
x=0
gt=np.linspace(0,t_final*g,steps)
gamma=0.1*g
kappa=np.linspace(0,2.5*g,5)#[0,0.1*g,0.2*g,0.3*g,0.4*g,0.5*g,0.6*g,0.7*g,0.8*g,0.9*g,g,1.1*g,1.2*g,1.3*g,1.4*g,1.5*g,1.6*g,1.7*g,1.8*g,1.9*g,2*g]

param=kappa
colors=mpl.colormaps['inferno'](np.linspace(0,1,len(param)+1))
lines_legend1=[]
lines_legend2=[]
labels_legend=[]

# coherencias_u=np.zeros((len(param),66,steps))
# coherencias_d=np.zeros((len(param),66,steps))
fg_u=np.zeros((len(param),steps))
fg_d=np.zeros((len(param),steps))
concu_u=np.zeros((len(param),steps))
concu_d=np.zeros((len(param),steps))
for i,k in enumerate(param):
    #,coherencias_u[i],coherencias_d[i]
    fg_u[i],fg_d[i],concu_u[i],concu_d[i]=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final=t_final,steps=steps)
    
fg_min=min(min(fg_u.flatten()),min(fg_d.flatten()))
fg_max=max(max(fg_u.flatten()),max(fg_d.flatten()))

'''--------PLOT-------'''
fig = plt.figure(1,(16,9))
fig.suptitle(f'$\Delta={d}$ $\chi = {x}$ $|\psi_0>$='+psi0Name)
ax1 = fig.add_subplot(211)  #fg unitario en solido y disipativo en rayado
ax2 = fig.add_subplot(212)  #concu unitario en solido y disipativo en rayado

for i,k in enumerate(param):
    line_fg_u,=ax1.plot(gt,fg_u[i],color=colors[i],linestyle='solid')
    labels_legend.append(f'U k={k/g}g')
    line_fg_d,=ax1.plot(gt,fg_d[i],color=colors[i],linestyle='dashed')
    # labels_legend.append(f'D k={k/g}g')
    lines_legend1.append(line_fg_u)
    # lines_legend1.append(line_fg_d)

    line_concu_u,=ax2.plot(gt,concu_u[i],color=colors[i],linestyle='solid')
    line_concu_d,=ax2.plot(gt,concu_d[i],color=colors[i],linestyle='dashed')
    lines_legend2.append(line_concu_u)
    # lines_legend2.append(line_concu_d)

ax1.legend(lines_legend1,labels_legend)
ax2.legend(lines_legend2,labels_legend)
plt.show()



# '''------Animacion-----'''
# anim_FG=anim_univsdis("FG",fg_u,fg_d,kappa,"k",t_final,steps,psi0Name,[0,g*t_final,fg_min,fg_max])
# anim_concu=anim_univsdis("Concu",concu_u,concu_d,kappa,"k",t_final,steps,psi0Name,[0,g*t_final,0,1])


# anim_FG.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} FG kappa uni vs disip chi={x/g}g delta={d/g}g.gif", writer='pillow')
# anim_concu.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} Concu kappa uni vs disip FG chi={x/g}g delta={d/g}g.gif", writer='pillow')