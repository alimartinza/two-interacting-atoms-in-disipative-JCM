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

from jcm_lib import plots_uni_vs_dis_chi,plots_uni_vs_dis_delta,plots_uni_vs_dis_gamma,plots_uni_vs_dis_J,plots_uni_vs_dis_kappa


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


script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

psi0=(ee0-gg2).unit()
psi0Name="eg0"
steps=50000
t_final=50000
w_0=1
g=0.001*w_0
d=0
x=0
J=0
gamma=0.1*g
p=0#0.05*gamma
kappa=0#np.linspace(0,2.5*g,5)#[0,0.1*g,0.2*g,0.3*g,0.4*g,0.5*g,0.6*g,0.7*g,0.8*g,0.9*g,g,1.1*g,1.2*g,1.3*g,1.4*g,1.5*g,1.6*g,1.7*g,1.8*g,1.9*g,2*g]

plots_uni_vs_dis_gamma(w_0=w_0, g=g, kappa=0.1*g, J=0, d=0, x=0, gamma=[0,0.1*g,0.5*g,g,2*g], p=p, psi0=psi0, psi0Name=psi0Name, t_final=t_final, steps=steps)
# plots_uni_vs_dis_chi(w_0=w_0, g=g, kappa=kappa, J=J, d=d, x=[0,0.1*g,g,3*g], gamma=gamma, p=p, psi0=psi0, psi0Name=psi0Name, t_final=t_final, steps=steps)
# plots_uni_vs_dis_delta(w_0=w_0, g=g, kappa=kappa, J=J, d=[-0.1*g,0,0.1*g,g,3*g], x=x, gamma=gamma, p=p, psi0=psi0, psi0Name=psi0Name, t_final=t_final, steps=steps)
# plots_uni_vs_dis_kappa(w_0=w_0, g=g, kappa=[0,0.1*g,g,2*g], J=0, d=-2*g, x=x, gamma=gamma, p=p, psi0=psi0, psi0Name=psi0Name, t_final=t_final, steps=steps)
# plots_uni_vs_dis_kappa(w_0=w_0, g=g, kappa=[0,0.1*g,g,2*g], J=0, d=0.1*g, x=x, gamma=gamma, p=p, psi0=psi0, psi0Name=psi0Name, t_final=t_final, steps=steps)
# '''------Animacion-----'''
# anim_FG=anim_univsdis("FG",fg_u,fg_d,kappa,"k",t_final,steps,psi0Name,[0,g*t_final,fg_min,fg_max])
# anim_concu=anim_univsdis("Concu",concu_u,concu_d,kappa,"k",t_final,steps,psi0Name,[0,g*t_final,0,1])


# anim_FG.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} FG kappa uni vs disip chi={x/g}g delta={d/g}g.gif", writer='pillow')
# anim_concu.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} Concu kappa uni vs disip FG chi={x/g}g delta={d/g}g.gif", writer='pillow')