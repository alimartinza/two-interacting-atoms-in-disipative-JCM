from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk
import pandas as pd


script_path=os.path.dirname(__file__)

folder_names=["8_27_20 disipativo lineal","8_28_0 disipativo bs","8_28_2 unitario lineal","8_28_3 unitario bs"]
condiciones_iniciales=["ee0","gg1","eg0"]
relative_path="datos"+"\\"+folder_names[0]+"\\"+condiciones_iniciales[0]

path=os.path.join(script_path, relative_path)
if os.path.exists(path):
    os.chdir(path)
else: 
    print("Dir %s does not exist", path)

w0=1
J=0
g=0.001*w0
k=0.1*g
p=0.005*g
x=0#[0,1/4*g,1/2*g]
d=0#[0,0.5*g,2*g]
gamma=0.1*g#[0.1*g,2*g]

g_str=str(g).replace('.','_')
k_str=str(k).replace('.','_')
J_str=str(J).replace('.','_')
d_str=str(d).replace('.','_')
x_str=str(x).replace('.','_')
gamma_str=str(gamma).replace('.','_')
p_str=str(p).replace('.','_')
t_final=100000
steps=100000
t=np.linspace(0,t_final,steps)
save_plot=True
plot_show=False

figname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'

data=pd.read_csv(csvname)

def respuesta_si():
    global disipation
    disipation = True
    root.destroy()

def respuesta_no():
    global disipation
    disipation = False
    root.destroy()

# Crear la ventana principal
root = tk.Tk()
root.title("Disipacion")

# Crear el mensaje y los botones
label = tk.Label(root, text="La simulacion tenia disipacion?", font=("Arial", 14))
label.pack(pady=20)

boton_si = tk.Button(root, text="Sí", command=respuesta_si, width=10)
boton_si.pack(side="left", padx=20, pady=20)

boton_no = tk.Button(root, text="No", command=respuesta_no, width=10)
boton_no.pack(side="right", padx=20, pady=20)

# Ejecutar la ventana
root.mainloop()

coherencias={'0,1':[],'0,2':[],'0,3':[],'0,4':[],'0,5':[],'0,6':[],'0,7':[],'0,8':[],'0,9':[],'0,10':[],'0,11':[],
                            '1,2':[],'1,3':[],'1,4':[],'1,5':[],'1,6':[],'1,7':[],'1,8':[],'1,9':[],'1,10':[],'1,11':[],
                                    '2,3':[],'2,4':[],'2,5':[],'2,6':[],'2,7':[],'2,8':[],'2,9':[],'2,10':[],'2,11':[],
                                            '3,4':[],'3,5':[],'3,6':[],'3,7':[],'3,8':[],'3,9':[],'3,10':[],'3,11':[],
                                                    '4,5':[],'4,6':[],'4,7':[],'4,8':[],'4,9':[],'4,10':[],'4,11':[],
                                                            '5,6':[],'5,7':[],'5,8':[],'5,9':[],'5,10':[],'5,11':[],
                                                                    '6,7':[],'6,8':[],'6,9':[],'6,10':[],'6,11':[],
                                                                            '7,8':[],'7,9':[],'7,10':[],'7,11':[],
                                                                                    '8,9':[],'8,10':[],'8,11':[],
                                                                                            '9,10':[],'9,11':[],
                                                                                                    '10,11':[]}

print(data['pr(ee0)'][1])
print(type(data['pr(ee0)'][1]))

coherenciasStartTime = time.process_time()
if not disipation:
    for i in range(len(data['sol states'])):
        for j in range(12): 
            for l in range(j+1,12):
                coherencias[str(j)+','+str(l)].append(data['sol states'][i][j]*data['sol states'][i][l])        
else:
    for j in range(12): 
        for l in range(j+1,12):
            c_help=np.zeros(len(data['sol states']),dtype='complex')
            for i in range(len(data['sol states'])):
                c_help[i]=data['sol states'][i][j][l]
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
ax[0].plot(g*t,data['pr(gg0)'],label=data.keys()[1],color='black')
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
ax[0].plot(g*t,data['pr(gg1)'],label=data.keys()[1],color='black')
ax[0].plot(g*t,data['pr(eg0+ge0)'],label=data.keys()[2],color='blue')
ax[0].plot(g*t,data['pr(eg0-ge0)'],label=data.keys()[3],color='red')
plot_coherencias(3,0) #N=1
ax[1].plot(g*t,data['pr(gg1)'],label=data.keys()[1],color='black')
ax[1].plot(g*t,data['pr(eg0+ge0)'],label=data.keys()[2],color='blue')
ax[1].plot(g*t,data['pr(eg0-ge0)'],label=data.keys()[3],color='red')
plot_coherencias(4,1) #N=1
ax[2].plot(g*t,data['pr(gg1)'],label=data.keys()[1],color='black')
ax[2].plot(g*t,data['pr(eg0+ge0)'],label=data.keys()[2],color='blue')
ax[2].plot(g*t,data['pr(eg0-ge0)'],label=data.keys()[3],color='red')
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
ax[0].plot(g*t,data['pr(gg2)'],label=data.keys()[4],color='black')
ax[0].plot(g*t,data['pr(eg1+ge1)'],label=data.keys()[5],color='blue')
ax[0].plot(g*t,data['pr(eg1-ge1)'],label=data.keys()[6],color='red')
ax[0].plot(g*t,data['pr(ee0)'],label=data.keys()[7],color='green')
plot_coherencias(0,0) #N=2

ax[1].plot(g*t,data['pr(gg2)'],label=data.keys()[4],color='black')
ax[1].plot(g*t,data['pr(eg1+ge1)'],label=data.keys()[5],color='blue')
ax[1].plot(g*t,data['pr(eg1-ge1)'],label=data.keys()[6],color='red')
ax[1].plot(g*t,data['pr(ee0)'],label=data.keys()[7],color='green')
plot_coherencias(5,1) #N=2

ax[2].plot(g*t,data['pr(gg2)'],label=data.keys()[4],color='black')
ax[2].plot(g*t,data['pr(eg1+ge1)'],label=data.keys()[5],color='blue')
ax[2].plot(g*t,data['pr(eg1-ge1)'],label=data.keys()[6],color='red')
ax[2].plot(g*t,data['pr(ee0)'],label=data.keys()[7],color='green')
ax[2].set_xlabel('gt')
plot_coherencias(6,2) #N=2 ESTA TIENE ALGUN PROBLEMA, SE GRAFICAN EL C(6,9) Y c(6,3) (CREO QUE ESOS) PERO DEBERIAN SER 0, Y SE GRAFICAN MUCHO NO ES ERROR NUMERICO

ax[3].plot(g*t,data['pr(gg2)'],label=data.keys()[4],color='black')
ax[3].plot(g*t,data['pr(eg1+ge1)'],label=data.keys()[5],color='blue')
ax[3].plot(g*t,data['pr(eg1-ge1)'],label=data.keys()[6],color='red')
ax[3].plot(g*t,data['pr(ee0)'],label=data.keys()[7],color='green')
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
ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')
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
plt.plot(g*t,data['1/2 <sz1+sz2>'],label=data.keys()[11],color='black')
plt.plot(g*t,data['<sx1>'],label=data.keys()[12],color='blue')
plt.plot(g*t,data['<sx2>'],label=data.keys()[13],color='red')
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
