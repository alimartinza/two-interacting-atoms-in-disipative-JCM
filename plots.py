from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk
import pandas as pd


figname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'


coherenciasStartTime = time.process_time()
if not disipation:
    for i in range(len(sol.states)):
        for j in range(12): 
            for l in range(j+1,12):
                coherencias[str(j)+','+str(l)].append(sol.states[i][j]*sol.states[i][l])        
else:
    for j in range(12): 
        for l in range(j+1,12):
            c_help=np.zeros(len(sol.states),dtype='complex')
            for i in range(len(sol.states)):
                c_help[i]=sol.states[i][j][l]
                coherencias[str(j)+','+str(l)].append(c_help[i])
coherenciasRunTime = time.process_time()-coherenciasStartTime

def plot_ReIm_coherencias(n:int,n_ax:int,xlabel=None,ylabel=None):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    colors = plt.cm.jet(np.linspace(0,1,12))
    i=0
    for key in coherencias.keys():
        if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                ax[n_ax].plot(g*t,np.real(coherencias[key]),linestyle='dashed',label=f'Re[C({key})]',color=colors[i])
                ax[n_ax].plot(g*t,np.imag(coherencias[key]),linestyle='dashdot',label=f'Im[C({key})]',color=colors[i])
                i+=1
    ax[n_ax].legend()
    ax[n_ax].set_xlabel(xlabel)
    ax[n_ax].set_ylabel(ylabel)

def plot_coherencias(n:int,n_ax:int,xlabel='gt',ylabel='Abs(Coh)'):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    colors = ['#000000','#000000','#000000','#ff7043','#000000','#000000','#000000','#000000','#000000','#1976d2','#4caf50','#000000'] #plt.cm.jet(np.linspace(0,1,12))
    i=0
    if n==1:
        for key in ['0,1','1,2','1,3','1,4','1,5','1,6','1,7','1,8','1,9','1,10','1,11']:
            ax[n_ax].plot(g*t,np.abs(coherencias[key]),linestyle='dashed',color=colors[i],label=f'C({key})')
            i+=1
    else:
        for key in coherencias.keys():
            if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                    ax[n_ax].plot(g*t,np.abs(coherencias[key]),linestyle='dashed',color=colors[i],label=f'C({key})')
                    i+=1
    ax[n_ax].legend()
    # ax[n_ax].set_xlabel(xlabel)
    ax[n_ax].set_ylabel(ylabel)
    
'''---------------PLOTS-----------------------'''

'''--- N=0 ---'''
fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
ax=[ax]
fig.suptitle('N=0')
ax[0].plot(g*t,data['pr(gg0)'],label=ops_nomb[0],color='black')
plot_coherencias(9,0) #N=0
ax[0].set_xlabel('gt')

if save_plot==True:
    plt.savefig(f'0\\{figname}',dpi=100)
else:
    None
if plot_show==True:
    plt.show()
else: 
    None    
plt.close()
'''--- N=1 ---'''
fig,ax=plt.subplots(3,1,figsize=(16, 9),sharex=True) 
fig.suptitle('N=1')
ax[0].plot(g*t,data['pr(gg1)'],label=ops_nomb[1],color='black')
ax[0].plot(g*t,data['pr(eg0+ge0)'],label=ops_nomb[2],color='blue')
ax[0].plot(g*t,data['pr(eg0-ge0)'],label=ops_nomb[3],color='red')
plot_coherencias(3,0) #N=1
ax[1].plot(g*t,data['pr(gg1)'],label=ops_nomb[1],color='black')
ax[1].plot(g*t,data['pr(eg0+ge0)'],label=ops_nomb[2],color='blue')
ax[1].plot(g*t,data['pr(eg0-ge0)'],label=ops_nomb[3],color='red')
plot_coherencias(4,1) #N=1
ax[2].plot(g*t,data['pr(gg1)'],label=ops_nomb[1],color='black')
ax[2].plot(g*t,data['pr(eg0+ge0)'],label=ops_nomb[2],color='blue')
ax[2].plot(g*t,data['pr(eg0-ge0)'],label=ops_nomb[3],color='red')
ax[2].set_xlabel('gt')
plot_coherencias(10,2) #N=1
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'1\\{figname}',dpi=100)
else: 
    None
plt.close()
'''--- N=2 ---'''
fig,ax=plt.subplots(2,2,figsize=(16, 9),tight_layout=True,sharex=True) 
ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
fig.suptitle('N=2')
ax[0].plot(g*t,data['pr(gg2)'],label=ops_nomb[4],color='black')
ax[0].plot(g*t,data['pr(eg1+ge1)'],label=ops_nomb[5],color='blue')
ax[0].plot(g*t,data['pr(eg1-ge1)'],label=ops_nomb[6],color='red')
ax[0].plot(g*t,data['pr(ee0)'],label=ops_nomb[7],color='green')
plot_coherencias(0,0) #N=2

ax[1].plot(g*t,data['pr(gg2)'],label=ops_nomb[4],color='black')
ax[1].plot(g*t,data['pr(eg1+ge1)'],label=ops_nomb[5],color='blue')
ax[1].plot(g*t,data['pr(eg1-ge1)'],label=ops_nomb[6],color='red')
ax[1].plot(g*t,data['pr(ee0)'],label=ops_nomb[7],color='green')
plot_coherencias(5,1) #N=2

ax[2].plot(g*t,data['pr(gg2)'],label=ops_nomb[4],color='black')
ax[2].plot(g*t,data['pr(eg1+ge1)'],label=ops_nomb[5],color='blue')
ax[2].plot(g*t,data['pr(eg1-ge1)'],label=ops_nomb[6],color='red')
ax[2].plot(g*t,data['pr(ee0)'],label=ops_nomb[7],color='green')
ax[2].set_xlabel('gt')
plot_coherencias(6,2) #N=2 ESTA TIENE ALGUN PROBLEMA, SE GRAFICAN EL C(6,9) Y c(6,3) (CREO QUE ESOS) PERO DEBERIAN SER 0, Y SE GRAFICAN MUCHO NO ES ERROR NUMERICO

ax[3].plot(g*t,data['pr(gg2)'],label=ops_nomb[4],color='black')
ax[3].plot(g*t,data['pr(eg1+ge1)'],label=ops_nomb[5],color='blue')
ax[3].plot(g*t,data['pr(eg1-ge1)'],label=ops_nomb[6],color='red')
ax[3].plot(g*t,data['pr(ee0)'],label=ops_nomb[7],color='green')
ax[3].set_xlabel('gt')
plot_coherencias(11,3) #N=2
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'2\\{figname}',dpi=100)
else: 
    None
plt.close()
'''--- N=3 ---'''

fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
ax=[ax]
fig.suptitle('N=3')
ax[0].plot(g*t,data['pr(eg2)'],label=ops_nomb[8],color='black')
ax[0].plot(g*t,data['pr(ge2)'],label=ops_nomb[9],color='blue')
ax[0].plot(g*t,data['pr(ee1)'],label=ops_nomb[10],color='red')
plot_coherencias(1,0) #N=3
plot_coherencias(7,0) #N=3
plot_coherencias(8,0) #N=3
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'3\\{figname}',dpi=100)
else: 
    None
plt.close()
'''--- VM Pauli ---'''
fig,ax=plt.subplots(1,1,figsize=(16, 9))
fig.suptitle('V.M. Pauli')
plt.plot(g*t,data['1/2 <sz1+sz2>'],label=ops_nomb[11],color='black')
plt.plot(g*t,data['<sx1>'],label=ops_nomb[12],color='blue')
plt.plot(g*t,data['<sx2>'],label=ops_nomb[13],color='red')
plt.legend()
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'pauli\\{figname}',dpi=100)
else: 
    None
plt.close()

'''--- Entropias ---'''
#PLOT PARA LAS ENTROPIAS
fig,ax=plt.subplots(2,1,figsize=(16, 9),sharex=True)
fig.suptitle("Entropia en A-A-F")
ax[0].plot(g*t,data['S von Neuman tot'],color='black')
# ax[0].set_xlabel('t')
ax[0].set_ylabel('S_vn')

ax[1].plot(g*t,data['S lineal tot'],color='red')
ax[1].set_xlabel('t')
ax[1].set_ylabel('S_lin')
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'entropia\\{figname}',dpi=100)
else: 
    None
plt.close()
#PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA SIMULACION ESTARIA COPADO

'''---Trazamos sobre el campo---'''
#Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS


#PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES
fig,ax=plt.subplots(3,1,figsize=(16, 9),sharex=True)
fig.suptitle("Sist. A-A sin foton")
ax[0].plot(g*t,data['S vN atom'],color='black')
# ax[0].set_xlabel('t')
ax[0].set_ylabel('S_vn')

ax[1].plot(g*t,data['S lin atom'],color='red')
# ax[1].set_xlabel('t')
ax[1].set_ylabel('S_lin')

ax[2].plot(g*t,data['Concu atom'],color='blue')
ax[2].set_xlabel('t')
ax[2].set_ylabel('Concurrence')
if plot_show==True:
    plt.show()
else: 
    None
if save_plot==True:
    plt.savefig(f'entropia_spin-spin\\{figname}',dpi=100)
else: 
    None
plt.close()
