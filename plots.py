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

# folder_names=["8_30_22 disipativo lineal","8_31_3 disipativo bs","8_31_8 unitario lineal","8_31_14 unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
# condiciones_iniciales=["ee0"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

#DEFINIMOS LOS PARAMETROS QUE NO VAMOS A QUERER MODIFICAR EN LOS GRAFICOS
w0=1
J=0
g=0.001*w0
k=0.1*g
p=0.005*g
# t_final=25000
# steps=2000
# t=np.linspace(0,t_final,steps)
save_plot=False
plot_show=True

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

def plot_ReIm_coherencias(data,n:int,ax,xlabel=None,ylabel=None):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    colors = plt.cm.jet(np.linspace(0,1,12))
    i=0
    for key in coherencias.keys():
        if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                ax.plot(g*data['t'],np.real(data[key]),linestyle='dashed',label=f'Re[C({key})]',color=colors[i])
                ax.plot(g*data['t'],np.imag(data[key]),linestyle='dashdot',label=f'Im[C({key})]',color=colors[i])
                i+=1
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_coherencias(data,n:int,ax,xlabel='gt',ylabel='Abs(Coh)'):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    cmap=mpl.colormaps["plasma"]
    colors=cmap(np.linspace(0,1,12))

    i=0
    if n==1:
        for key in ['0,1','1,2','1,3','1,4','1,5','1,6','1,7','1,8','1,9','1,10','1,11']:
            ax.plot(g*data['t'],np.abs(data[key]),linestyle='dashed',color=colors[i],alpha=0.5)#,label=f'C({key})')
            i+=1
    else:
        for key in coherencias.keys():
            print(key)
            if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                    print(type(data[key][0]))
                    ax.plot(g*data['t'],np.abs(data[key]),linestyle='dashed',color=colors[i],alpha=0.5)#,label=f'C({key})')
                    i+=1
    # ax.legend()
    # # ax[n_ax].set_xlabel(xlabel)
    # ax[n_ax].set_ylabel(ylabel)

def plot3D_gamma(condiciones_iniciales:list):
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
    folder_names=["disipativo lineal","disipativo bs","unitario lineal","unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # condiciones_iniciales=["eg0"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

    #PARA CADA CONDICION INICIAL HACEMOS LOS GRAFICOS, HACEMOS ITERACIONES PARA CADA CARPETA ASI COMPARAMOS LOS MODELOS 
    for ci in condiciones_iniciales:
        for folder_names in folder_names:

            relative_path="datos"+"\\"+folder_names+"\\"+ci 
            path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
            if os.path.exists(path):
                os.chdir(path)
            else: 
                print("Dir %s does not exist", path)

            #POR AHORA LOS PARAMETROS VAN A SER MANUALES, Y DEBERIAN SER LOS MISMOS QUE USAMOS EN LA SIMULACION. YO POR AHORA LA SIMULACION LARGA
            #LA HICE CON LOS PARAMETROS x=[0,1/4*g,1/2*g], d=[0,0.5*g,2*g], gamma=[0.1*g,2*g] ASI QUE CREO QUE ESOS VAN A QUEDAR ASI POR UN BUEN RATO
            x=0#[0,1/4*g,1/2*g]
            d=0#[0,0.5*g,2*g]
            gamma=[0.1*g,2*g] 
            colors=['red','black','blue','green']

            '''-------LAYOUT PARA LOS GRAFICOS------'''
            #PARA CADA GRAFICO QUE VAMOS A HACER, CREAMOS LA FIGURA EN UNA PRIMERA INSTANCIA ASI QUEDAN ESTATICOS, Y DESPUES HACEMOS UN LOOP POR LOS ARCHIVOS QUE VAN A ESTAR
            #INCLUIDOS EN CADA UNO PARA HACER LA COMPARACION
            '''CHECK TR=1'''
            fig_check=plt.figure(figsize=(16,9))
            fig_check.suptitle('Check TR=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_check = fig_check.add_subplot()
            # ax_check.set_yticks(np.array(gamma)/g)
            ax_check.set_xlabel('gt')
            # ax_check.set_ylabel('$\\gamma$/g')
            ax_check.set_ylabel('TR(RHO)')
            # ax_check.view_init(30,-40,0)
            # ax_check.set_yticks([0,1],np.array(gamma)/g)
            # ax_check.set_zlim(0,1)

            '''N=0'''
            fig0 = plt.figure(figsize=(16,9))
            fig0.suptitle('N=0 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax0 = fig0.add_subplot(projection='3d')
            ax0.set_yticks(np.array(gamma)/g)
            ax0.set_xlabel('gt')
            ax0.set_ylabel('$\\gamma$/g')
            ax0.set_zlabel('Amp. Prob. ')
            ax0.view_init(30,-40,0)
            ax0.set_yticks([0,1],np.array(gamma)/g)
            ax0.set_zlim(0,1)

            '''N=1'''
            fig1 = plt.figure(figsize=(16,9))
            ax1 = fig1.add_subplot(projection='3d')
            fig1.suptitle('N=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax1.set_xlabel('gt')
            ax1.set_ylabel('$\\gamma$/g')
            ax1.set_zlabel('Amp. Prob. ')
            ax1.set_yticks([0,1],np.array(gamma)/g)
            ax1.view_init(30,-40,0)
            ax1.set_yticks([0,1],np.array(gamma)/g)
            ax1.set_zlim(0,1)
            '''N=2'''
            fig2 = plt.figure(figsize=(16,9))
            ax2 = fig2.add_subplot(projection='3d')
            fig2.suptitle('N=2 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax2.set_xlabel('gt')
            ax2.set_ylabel('$\\gamma$/g')
            ax2.set_zlabel('Amp. Prob. ')
            ax2.set_yticks([0,1],np.array(gamma)/g)
            ax2.view_init(30,-40,0)
            ax2.set_yticks([0,1],np.array(gamma)/g)
            ax2.set_zlim(0,1)
            '''PAULI'''
            fig_pauli = plt.figure(figsize=(16,9))
            ax_pauli = fig_pauli.add_subplot(projection='3d')
            fig_pauli.suptitle('Pauli '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_pauli.set_xlabel('gt')
            ax_pauli.set_ylabel('$\\gamma$/g')
            ax_pauli.set_zlabel('V.M.')
            ax_pauli.set_yticks([0,1],np.array(gamma)/g)
            ax_pauli.view_init(30,-40,0)
            ax_pauli.set_yticks([0,1],np.array(gamma)/g)
            ax_pauli.set_zlim(-1,1)
            '''ENTROPIA VON NEUMAN Y LINEAL'''
            fig_S = plt.figure(figsize=(16,9))
            ax_Slin = fig_S.add_subplot(121,projection='3d')
            ax_Svn = fig_S.add_subplot(122,projection='3d')
            fig_S.suptitle('Entropia '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Svn.set_zlabel('S')
            ax_Slin.set_xlabel('gt')
            ax_Slin.set_ylabel('$\\gamma$/g')
            ax_Svn.set_ylabel('$\\gamma$/g')
            ax_Svn.set_xlabel('gt')
            ax_Svn.view_init(30,-40,0)
            ax_Svn.set_yticks([0,1],np.array(gamma)/g)
            ax_Svn.set_zlim(0,np.log(8))
            ax_Slin.view_init(30,-40,0)
            ax_Slin.set_yticks([0,1],np.array(gamma)/g)
            ax_Slin.set_zlim(0,1-1/8)

            '''ESTADO REDUCIDO: ENTROPIA Y CONCURRENCIA'''
            fig_Sr = plt.figure(figsize=(16,9))
            ax_Srlin = fig_Sr.add_subplot(131,projection='3d')
            ax_Srvn = fig_Sr.add_subplot(132,projection='3d')
            ax_Con = fig_Sr.add_subplot(133,projection='3d')
            fig_Sr.suptitle('Entropia Reducida '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Srvn.set_zlabel('S')
            ax_Srlin.set_zlabel('S')
            ax_Con.set_zlabel('C')
            ax_Con.set_ylabel('$\\gamma$/g')
            ax_Srlin.set_ylabel('$\\gamma$/g')
            ax_Srvn.set_ylabel('$\\gamma$/g')
            ax_Con.set_xlabel('gt')
            ax_Srlin.set_xlabel('gt')
            ax_Srvn.set_xlabel('gt')
            ax_Srvn.view_init(30,-40,0)
            ax_Srvn.set_yticks([0,1],np.array(gamma)/g)
            ax_Srvn.set_zlim(0,np.log(4))
            ax_Srlin.view_init(30,-40,0)
            ax_Srlin.set_yticks([0,1],np.array(gamma)/g)
            ax_Srlin.set_zlim(0,1-1/4)
            ax_Con.view_init(30,-40,0)
            ax_Con.set_yticks([0,1],np.array(gamma)/g)
            ax_Con.set_zlim(0,1)
    
            #AHORA HACEMOS EL LOOP ENTRE LOS ARCHIVOS DE DIFERENTES PARAMETROS Y LOS PONEMOS EN SU CORRESPONDIENTE GRAFICO Y EJE
            for i,gamma_m in enumerate(gamma):
                g_str=str(g).replace('.','_')
                k_str=str(k).replace('.','_')
                J_str=str(J).replace('.','_')
                d_str=str(d).replace('.','_')
                x_str=str(x).replace('.','_')
                gamma_m_str=str(gamma_m).replace('.','_')
                p_str=str(p).replace('.','_')
                
                param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_m_str} p={p_str}'
                csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_m_str} p={p_str}.csv'
                
                data=pd.read_csv(csvname,header=0,index_col=0)

                # dataKeys=np.loadtxt(csvname,dtype='object',delimiter=',',max_rows=1)
                # print(dataKeys)
                # data={}
                # for i,keys in enumerate(dataKeys):
                #     try:
                #         print(str(keys))
                #         data[str(keys)]=np.loadtxt(csvname,dtype='complex',delimiter=',',skiprows=1,usecols=i)
                #     except TypeError as e:
                #         print(e)
                

                '''----DATOS DE LOS PLOTS----'''
                '''CHECK TR=1'''
                trace=data['pr(gg0)']+data['pr(gg1)']+data['pr(gg2)']+data['pr(ee0)']+data['pr(eg0+ge0)']+data['pr(ge0-eg0)']+data['pr(eg1+ge1)']+data['pr(eg1-ge1)']
                ax_check.plot(g*data['t'],trace,color=colors[i], alpha=0.8)
                '''--- N=0 ---'''
                line0,=ax0.plot(g*data['t'], data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],[data.keys()[0]])
                # plot_coherencias(data,9,ax0)#,0) #N=0
                
                '''--- N=1 ---'''
                line11,=ax1.plot(g*data['t'],data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8,zorder=10-i)
                line12,=ax1.plot(g*data['t'],data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*data['t'],data['pr(ge0-eg0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                # plot_coherencias(data,3,ax1) #N=1
                # plot_coherencias(data,6,ax1) #N=1
                # plot_coherencias(data,10,ax1) #N=1
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*data['t'],data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*data['t'],data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*data['t'],data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*data['t'],data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                # plot_coherencias(data,0,ax2) #N=2
                # plot_coherencias(data,4,ax2) #N=2
                # plot_coherencias(data,7,ax2) #N=2 
                # plot_coherencias(data,11,ax2) #N=2
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*data['t'],data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*data['t'],data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*data['t'],data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*data['t'],data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*data['t'],data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*data['t'],data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*data['t'],data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*data['t'],data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*data['t'],data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*data['t'],data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*data['t'],data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_Srvn.legend([lineSrvn,lineSrlin,lineCon],['S_vN','S_lin','Conc'])
            
            script_path=os.path.dirname(__file__)            
            relative_path="graficos resumen"+"\\"+ci+"\\"+"gamma"
            path=os.path.join(script_path, relative_path)
            if os.path.exists(path):
                os.chdir(path)
            else: 
                os.makedirs(path)
                os.chdir(path)

            fig0.savefig(ci+' n=0 '+folder_names,dpi=100)
            fig2.savefig(ci+' n=2 '+folder_names,dpi=100)
            fig1.savefig(ci+' n=1 '+folder_names,dpi=100)
            fig_pauli.savefig(ci+' pauli '+folder_names,dpi=100)
            fig_S.savefig(ci+' entropia '+folder_names,dpi=100)
            fig_Sr.savefig(ci+' entropia reducida '+folder_names,dpi=100)
            plt.close()

def plot3D_x(condiciones_iniciales:list):
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
    folder_names=["disipativo lineal","disipativo bs","unitario lineal","unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # condiciones_iniciales=["ee0"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

    #PARA CADA CONDICION INICIAL HACEMOS LOS GRAFICOS, HACEMOS ITERACIONES PARA CADA CARPETA ASI COMPARAMOS LOS MODELOS 
    for ci in condiciones_iniciales:
        for folder_names in folder_names:

            relative_path="datos"+"\\"+folder_names+"\\"+ci 
            path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
            if os.path.exists(path):
                os.chdir(path)
            else: 
                print("Dir %s does not exist", path)

            #POR AHORA LOS PARAMETROS VAN A SER MANUALES, Y DEBERIAN SER LOS MISMOS QUE USAMOS EN LA SIMULACION. YO POR AHORA LA SIMULACION LARGA
            #LA HICE CON LOS PARAMETROS x=[0,1/4*g,1/2*g], d=[0,0.5*g,2*g], gamma=[0.1*g,2*g] ASI QUE CREO QUE ESOS VAN A QUEDAR ASI POR UN BUEN RATO
            x=[0,1/4*g,1/2*g]
            d=0#[0,0.5*g,2*g]
            gamma=0.1*g#[0.1*g,2*g] 
            colors=['red','black','blue','green']

            '''-------LAYOUT PARA LOS GRAFICOS------'''
            #PARA CADA GRAFICO QUE VAMOS A HACER, CREAMOS LA FIGURA EN UNA PRIMERA INSTANCIA ASI QUEDAN ESTATICOS, Y DESPUES HACEMOS UN LOOP POR LOS ARCHIVOS QUE VAN A ESTAR
            #INCLUIDOS EN CADA UNO PARA HACER LA COMPARACION
            '''N=0'''
            fig0 = plt.figure(figsize=(16,9))
            fig0.suptitle('N=0 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax0 = fig0.add_subplot(projection='3d')
            ax0.set_yticks(np.array(x)/g)
            ax0.set_xlabel('gt')
            ax0.set_ylabel('$\\chi$/g')
            ax0.set_zlabel('Amp. Prob. ')
            ax0.view_init(30,-40,0)
            ax0.set_yticks([0,1,2],np.array(x)/g)
            ax0.set_zlim(0,1)

            '''N=1'''
            fig1 = plt.figure(figsize=(16,9))
            ax1 = fig1.add_subplot(projection='3d')
            fig1.suptitle('N=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax1.set_xlabel('gt')
            ax1.set_ylabel('$\\chi$/g')
            ax1.set_zlabel('Amp. Prob. ')
            ax1.set_yticks(np.array(x)/g)
            ax1.view_init(30,-40,0)
            ax1.set_yticks([0,1,2],np.array(x)/g)
            ax1.set_zlim(0,1)

            '''N=2'''
            fig2 = plt.figure(figsize=(16,9))
            ax2 = fig2.add_subplot(projection='3d')
            fig2.suptitle('N=2 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax2.set_xlabel('gt')
            ax2.set_ylabel('$\\chi$/g')
            ax2.set_zlabel('Amp. Prob. ')
            ax2.set_yticks([0,1,2],np.array(x)/g)
            ax2.view_init(30,-40,0)
            ax2.set_yticks([0,1,2],np.array(x)/g)
            # ax2.set_zlim(0,1)

            '''PAULI'''
            fig_pauli = plt.figure(figsize=(16,9))
            ax_pauli = fig_pauli.add_subplot(projection='3d')
            fig_pauli.suptitle('Pauli '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_pauli.set_xlabel('gt')
            ax_pauli.set_ylabel('$\\chi$/g')
            ax_pauli.set_zlabel('V.M.')
            ax_pauli.set_yticks([0,1,2],np.array(x)/g)
            ax_pauli.view_init(30,-40,0)
            ax_pauli.set_yticks([0,1,2],np.array(x)/g)
            ax_pauli.set_zlim(-1,1)

            '''ENTROPIA VON NEUMAN Y LINEAL'''
            fig_S = plt.figure(figsize=(16,9))
            ax_Slin = fig_S.add_subplot(121,projection='3d')
            ax_Svn = fig_S.add_subplot(122,projection='3d')
            fig_S.suptitle('Entropia '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Svn.set_zlabel('S')
            ax_Slin.set_xlabel('gt')
            ax_Slin.set_ylabel('$\\chi$/g')
            ax_Svn.set_ylabel('$\\chi$/g')
            ax_Svn.set_xlabel('gt')
            ax_Svn.view_init(30,-40,0)
            ax_Svn.set_yticks([0,1,2],np.array(x)/g)
            ax_Svn.set_zlim(0,np.log(8))
            ax_Slin.view_init(30,-40,0)
            ax_Slin.set_yticks([0,1,2],np.array(x)/g)
            ax_Slin.set_zlim(0,np.log(8))
    

            '''ESTADO REDUCIDO: ENTROPIA Y CONCURRENCIA'''
            fig_Sr = plt.figure(figsize=(16,9))
            ax_Srlin = fig_Sr.add_subplot(131,projection='3d')
            ax_Srvn = fig_Sr.add_subplot(132,projection='3d')
            ax_Con = fig_Sr.add_subplot(133,projection='3d')
            fig_Sr.suptitle('Entropia Reducida '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Srvn.set_zlabel('S')
            ax_Srlin.set_zlabel('S')
            ax_Con.set_zlabel('C')
            ax_Con.set_ylabel('$\\chi$/g')
            ax_Srlin.set_ylabel('$\\chi$/g')
            ax_Srvn.set_ylabel('$\\chi$/g')
            ax_Con.set_xlabel('gt')
            ax_Srlin.set_xlabel('gt')
            ax_Srvn.set_xlabel('gt')
            ax_Srvn.view_init(30,-40,0)
            ax_Srvn.set_yticks([0,1,2],np.array(x)/g)
            ax_Srvn.set_zlim(0,np.log(4))
            ax_Srlin.view_init(30,-40,0)
            ax_Srlin.set_yticks([0,1,2],np.array(x)/g)
            ax_Srlin.set_zlim(0,np.log(4))
            ax_Con.view_init(30,-40,0)
            ax_Con.set_yticks([0,1,2],np.array(x)/g)
            ax_Con.set_zlim(0,1)
    
            #AHORA HACEMOS EL LOOP ENTRE LOS ARCHIVOS DE DIFERENTES PARAMETROS Y LOS PONEMOS EN SU CORRESPONDIENTE GRAFICO Y EJE
            for i,x in enumerate(x):
                g_str=str(g).replace('.','_')
                k_str=str(k).replace('.','_')
                J_str=str(J).replace('.','_')
                d_str=str(d).replace('.','_')
                x_str=str(x).replace('.','_')
                gamma_str=str(gamma).replace('.','_')
                p_str=str(p).replace('.','_')
                
                param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
                csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
                
                data=pd.read_csv(csvname,header=0,index_col=0)

                '''----DATOS DE LOS PLOTS----'''

                '''--- N=0 ---'''
                line0,=ax0.plot(g*data['t'], data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],[data.keys()[0]])
                # plot_coherencias(data,9,ax0)#,0) #N=0                

                
                '''--- N=1 ---'''
                line11,=ax1.plot(g*data['t'],data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line12,=ax1.plot(g*data['t'],data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*data['t'],data['pr(ge0-eg0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                # plot_coherencias(data,3,ax1) #N=1
                # plot_coherencias(data,6,ax1) #N=1
                # plot_coherencias(data,10,ax1) #N=1
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*data['t'],data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*data['t'],data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*data['t'],data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*data['t'],data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                # plot_coherencias(data,0,ax2) #N=2
                # plot_coherencias(data,4,ax2) #N=2
                # plot_coherencias(data,7,ax2) #N=2 
                # plot_coherencias(data,11,ax2) #N=2
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*data['t'],data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*data['t'],data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*data['t'],data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*data['t'],data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*data['t'],data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*data['t'],data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*data['t'],data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*data['t'],data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_Srvn.legend([lineSrvn,lineSrlin,lineCon],['S_vN','S_lin','Conc'])
            
            script_path=os.path.dirname(__file__)            
            relative_path="graficos resumen"+"\\"+ci+"\\"+"x"
            path=os.path.join(script_path, relative_path)
            if os.path.exists(path):
                os.chdir(path)
            else: 
                os.makedirs(path)
                os.chdir(path)

            fig0.savefig(ci+' n=0 '+folder_names,dpi=100)
            fig2.savefig(ci+' n=2 '+folder_names,dpi=100)
            fig1.savefig(ci+' n=1 '+folder_names,dpi=100)
            fig_pauli.savefig(ci+' pauli '+folder_names,dpi=100)
            fig_S.savefig(ci+' entropia '+folder_names,dpi=100)
            fig_Sr.savefig(ci+' entropia reducida '+folder_names,dpi=100)
            plt.close()

def plot3D_delta(condiciones_iniciales:list):
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
    folder_names=["disipativo lineal","disipativo bs","unitario lineal","unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # folder_names=["9_7_9 disipativo lineal","9_7_9 disipativo bs","9_7_10 unitario lineal","9_7_11 unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # condiciones_iniciales=["w2"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

    #PARA CADA CONDICION INICIAL HACEMOS LOS GRAFICOS, HACEMOS ITERACIONES PARA CADA CARPETA ASI COMPARAMOS LOS MODELOS 
    for ci in condiciones_iniciales:
        for folder_names in folder_names:

            relative_path="datos"+"\\"+folder_names+"\\"+ci 
            path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
            if os.path.exists(path):
                os.chdir(path)
            else: 
                print("Dir %s does not exist", path)

            #POR AHORA LOS PARAMETROS VAN A SER MANUALES, Y DEBERIAN SER LOS MISMOS QUE USAMOS EN LA SIMULACION. YO POR AHORA LA SIMULACION LARGA
            #LA HICE CON LOS PARAMETROS x=[0,1/4*g,1/2*g], d=[0,0.5*g,2*g], gamma=[0.1*g,2*g] ASI QUE CREO QUE ESOS VAN A QUEDAR ASI POR UN BUEN RATO
            x=1/2*g#[0,1/4*g,1/2*g]
            d=[0,0.5*g,g,2*g]
            gamma=0.1*g#[0.1*g,2*g] 
            colors=['red','black','blue','green']

            '''-------LAYOUT PARA LOS GRAFICOS------'''
            #PARA CADA GRAFICO QUE VAMOS A HACER, CREAMOS LA FIGURA EN UNA PRIMERA INSTANCIA ASI QUEDAN ESTATICOS, Y DESPUES HACEMOS UN LOOP POR LOS ARCHIVOS QUE VAN A ESTAR
            #INCLUIDOS EN CADA UNO PARA HACER LA COMPARACION
            '''N=0'''
            fig0 = plt.figure(figsize=(16,9))
            fig0.suptitle('N=0 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax0 = fig0.add_subplot(projection='3d')
            ax0.set_yticks(np.array(d)/g)
            ax0.set_xlabel('gt')
            ax0.set_ylabel('$\\Delta$/g')
            ax0.set_zlabel('Amp. Prob. ')
            ax0.view_init(30,-40,0)
            ax0.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax0.set_zlim(0,1)
            '''N=1'''
            fig1 = plt.figure(figsize=(16,9))
            ax1 = fig1.add_subplot(projection='3d')
            fig1.suptitle('N=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax1.set_xlabel('gt')
            ax1.set_ylabel('$\\Delta$/g')
            ax1.set_zlabel('Amp. Prob. ')
            ax1.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax1.view_init(30,-40,0)
            ax1.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax1.set_zlim(0,1)

            '''N=2'''
            fig2 = plt.figure(figsize=(16,9))
            ax2 = fig2.add_subplot(projection='3d')
            fig2.suptitle('N=2 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax2.set_xlabel('gt')
            ax2.set_ylabel('$\\Delta$/g')
            ax2.set_zlabel('Amp. Prob. ')
            ax2.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax2.view_init(30,-40,0)
            ax2.set_yticks(np.arange(len(d)),np.array(d)/g)
            # ax2.set_zlim(0,1)

            '''PAULI'''
            fig_pauli = plt.figure(figsize=(16,9))
            ax_pauli = fig_pauli.add_subplot(projection='3d')
            fig_pauli.suptitle('Pauli '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_pauli.set_xlabel('gt')
            ax_pauli.set_ylabel('$\\Delta$/g')
            ax_pauli.set_zlabel('V.M.')
            ax_pauli.set_yticks(np.array(d)/g)
            ax_pauli.view_init(30,-40,0)
            ax_pauli.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_pauli.set_zlim(-1,1)

            '''ENTROPIA VON NEUMAN Y LINEAL'''
            fig_S = plt.figure(figsize=(16,9))
            ax_Slin = fig_S.add_subplot(121,projection='3d')
            ax_Svn = fig_S.add_subplot(122,projection='3d')
            fig_S.suptitle('Entropia A-A-F '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Svn.set_zlabel('S')
            ax_Slin.set_zlabel('S')
            ax_Slin.set_xlabel('gt')
            ax_Slin.set_ylabel('$\\Delta$/g')
            ax_Svn.set_ylabel('$\\Delta$/g')
            ax_Svn.set_xlabel('gt')
            ax_Svn.view_init(30,-40,0)
            ax_Slin.view_init(30,-40,0)
            ax_Svn.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_Slin.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_Svn.set_zlim(0,np.log(8))
            ax_Slin.set_zlim(0,1)
    
            '''ESTADO REDUCIDO: ENTROPIA Y CONCURRENCIA'''
            fig_Sr = plt.figure(figsize=(16,9))
            ax_Srlin = fig_Sr.add_subplot(131,projection='3d')
            ax_Srvn = fig_Sr.add_subplot(132,projection='3d')
            ax_Con = fig_Sr.add_subplot(133,projection='3d')
            fig_Sr.suptitle('Entropia Reducida '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Srvn.set_zlabel('S')
            ax_Srlin.set_zlabel('S')
            ax_Con.set_zlabel('C')
            ax_Con.set_ylabel('$\\Delta$/g')
            ax_Srlin.set_ylabel('$\\Delta$/g')
            ax_Srvn.set_ylabel('$\\Delta$/g')
            ax_Con.set_xlabel('gt')
            ax_Srlin.set_xlabel('gt')
            ax_Srvn.set_xlabel('gt')
            ax_Srvn.view_init(30,-40,0)
            ax_Srlin.view_init(30,-40,0)
            ax_Con.view_init(30,-40,0)
            ax_Srvn.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_Srlin.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_Con.set_yticks(np.arange(len(d)),np.array(d)/g)
            ax_Srvn.set_zlim(0,np.log(8))
            ax_Srlin.set_zlim(0,1)
            ax_Con.set_zlim(0,1)

    
            #AHORA HACEMOS EL LOOP ENTRE LOS ARCHIVOS DE DIFERENTES PARAMETROS Y LOS PONEMOS EN SU CORRESPONDIENTE GRAFICO Y EJE
            for i,d in enumerate(d):
                g_str=str(g).replace('.','_')
                k_str=str(k).replace('.','_')
                J_str=str(J).replace('.','_')
                d_str=str(d).replace('.','_')
                x_str=str(x).replace('.','_')
                gamma_str=str(gamma).replace('.','_')
                p_str=str(p).replace('.','_')
                
                param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
                csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
                
                data=pd.read_csv(csvname,header=0)

                '''----DATOS DE LOS PLOTS----'''

                '''--- N=0 ---'''
                line0,=ax0.plot(g*data['t'], data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],['gg0'])
                # plot_coherencias(data,9,ax0)#,0) #N=0

      
                '''--- N=1 ---'''
                line11,=ax1.plot(g*data['t'],data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line12,=ax1.plot(g*data['t'],data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*data['t'],data['pr(eg0-ge0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                # plot_coherencias(data,3,ax1) #N=1
                # plot_coherencias(data,6,ax1) #N=1
                # plot_coherencias(data,10,ax1) #N=1
                ax1.legend([line11,line12,line13],['gg1','eg0+ge0','eg0-ge0'])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*data['t'],data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*data['t'],data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*data['t'],data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*data['t'],data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                # plot_coherencias(data,0,ax2) #N=2
                # plot_coherencias(data,4,ax2) #N=2
                # plot_coherencias(data,7,ax2) #N=2 
                # plot_coherencias(data,11,ax2) #N=2
                ax2.legend([line21,line22,line23,line24],['gg2','eg1+ge1','eg1-ge1','ee0'])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*data['t'],data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*data['t'],data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*data['t'],data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],['$\\frac{1}{2}<\\sigma_z^{(1)}+\\sigma_z^{(2)}>$','$<\\sigma_x^{(1)}>$','$<\\sigma_x^{(2)}>$'])#[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*data['t'],data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*data['t'],data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*data['t'],data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*data['t'],data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*data['t'],data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_Srvn.legend([lineSrvn,lineSrlin,lineCon],['S_vN','S_lin','Conc'])


            script_path=os.path.dirname(__file__)            
            relative_path="graficos resumen"+"\\"+ci+"\\"+"delta"
            path=os.path.join(script_path, relative_path)
            if os.path.exists(path):
                os.chdir(path)
            else: 
                os.makedirs(path)
                os.chdir(path)

            fig0.savefig(ci+' n=0 '+folder_names,dpi=100)
            fig2.savefig(ci+' n=2 '+folder_names,dpi=100)
            fig1.savefig(ci+' n=1 '+folder_names,dpi=100)
            fig_pauli.savefig(ci+' pauli '+folder_names,dpi=100)
            fig_S.savefig(ci+' entropia '+folder_names,dpi=100)
            fig_Sr.savefig(ci+' entropia reducida '+folder_names,dpi=100)
            plt.close()

def plot2D_delta(cis:list,savePlots:bool=False,showPlots:bool=True):
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
    folder_names=["disipativo lineal","disipativo bs","unitario lineal","unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # folder_names=["9_7_9 disipativo lineal","9_7_9 disipativo bs","9_7_10 unitario lineal","9_7_11 unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    # condiciones_iniciales=["w2"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

    #PARA CADA CONDICION INICIAL HACEMOS LOS GRAFICOS, HACEMOS ITERACIONES PARA CADA CARPETA ASI COMPARAMOS LOS MODELOS 
    for ci in cis:
        for folder_names in folder_names:

            relative_path="datos"+"\\"+folder_names+"\\"+ci 
            path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
            if os.path.exists(path):
                os.chdir(path)
            else: 
                print("Dir %s does not exist", path)

            #POR AHORA LOS PARAMETROS VAN A SER MANUALES, Y DEBERIAN SER LOS MISMOS QUE USAMOS EN LA SIMULACION. YO POR AHORA LA SIMULACION LARGA
            #LA HICE CON LOS PARAMETROS x=[0,1/4*g,1/2*g], d=[0,0.5*g,2*g], gamma=[0.1*g,2*g] ASI QUE CREO QUE ESOS VAN A QUEDAR ASI POR UN BUEN RATO
            x=1/2*g#[0,1/4*g,1/2*g]
            d=[0,0.5*g,g,2*g]
            len_d=len(d)
            gamma=0.1*g#[0.1*g,2*g] 
            cmap1=mpl.colormaps["plasma"]
            colors1=cmap1(np.linspace(0,1,len_d))
            colors2=cmap1(np.linspace(0,1,2*len_d))
            colors3=cmap1(np.linspace(0,1,3*len_d))
            colors4=cmap1(np.linspace(0,1,4*len_d))

            colors_eval=[mpl.colormaps["Purples"](np.linspace(0,1,12)),mpl.colormaps["Blues"](np.linspace(0,1,12)),mpl.colormaps["Greens"](np.linspace(0,1,12)),mpl.colormaps["Oranges"](np.linspace(0,1,12))]

            '''-------LAYOUT PARA LOS GRAFICOS------'''
            #PARA CADA GRAFICO QUE VAMOS A HACER, CREAMOS LA FIGURA EN UNA PRIMERA INSTANCIA ASI QUEDAN ESTATICOS, Y DESPUES HACEMOS UN LOOP POR LOS ARCHIVOS QUE VAN A ESTAR
            #INCLUIDOS EN CADA UNO PARA HACER LA COMPARACION
            '''N=0'''
            fig0 = plt.figure(figsize=(16,9))
            fig0.suptitle('N=0 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax0 = fig0.add_subplot()
            ax0.set_yticks(np.array(d)/g)
            ax0.set_xlabel('gt')
            ax0.set_ylabel('Amp. Prob. ')
            ax0.set_ylim(0,1)
            '''N=1'''
            fig1 = plt.figure(figsize=(16,9))
            ax1 = fig1.add_subplot()
            fig1.suptitle('N=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax1.set_xlabel('gt')
            ax1.set_ylabel('Amp. Prob. ')
            ax1.set_ylim(0,1)

            '''N=2'''
            fig2 = plt.figure(figsize=(16,9))
            ax2 = fig2.add_subplot()
            fig2.suptitle('N=2 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax2.set_xlabel('gt')
            ax2.set_ylabel('Amp. Prob. ')
            # ax2.set_zlim(0,1)

            '''PAULI'''
            fig_pauli = plt.figure(figsize=(16,9))
            ax_pauli = fig_pauli.add_subplot()
            fig_pauli.suptitle('Pauli '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_pauli.set_xlabel('gt')
            ax_pauli.set_ylabel('V.M.')
            ax_pauli.set_ylim(-1,1)
            pauli_lines=[]
            pauli_names=[]

            '''ENTROPIA VON NEUMAN Y LINEAL'''
            fig_S = plt.figure(figsize=(16,9))
            ax_Slin = fig_S.add_subplot(121)
            ax_Svn = fig_S.add_subplot(122)
            fig_S.suptitle('Entropia A-A-F '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Svn.set_ylabel('S')
            ax_Slin.set_ylabel('S')
            ax_Slin.set_xlabel('gt')
            ax_Svn.set_xlabel('gt')
            ax_Svn.set_ylim(0,np.log(8))
            ax_Slin.set_ylim(0,1)

            '''ESTADO REDUCIDO: ENTROPIA Y CONCURRENCIA'''
            fig_Sr = plt.figure(figsize=(16,9))
            ax_Srlin = fig_Sr.add_subplot(131)
            ax_Srvn = fig_Sr.add_subplot(132)
            ax_Con = fig_Sr.add_subplot(133)
            fig_Sr.suptitle('Entropia Reducida '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_Srvn.set_ylabel('S')
            ax_Srlin.set_ylabel('S')
            ax_Con.set_ylabel('C')
            ax_Con.set_xlabel('gt')
            ax_Srlin.set_xlabel('gt')
            ax_Srvn.set_xlabel('gt')
            ax_Srvn.set_ylim(0,np.log(8))
            ax_Srlin.set_ylim(0,1)
            ax_Con.set_ylim(0,1)

            '''----Autovalores----'''
            fig_autoval=plt.figure()
            ax_eval=fig_autoval.add_subplot()
            ax_eval.set_xlabel('gt')
            ax_eval.set_ylabel('Eval')
            

            fig_fg=plt.figure()
            fig_fg.suptitle("Fase Geometrica "+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax_fg=fig_fg.add_subplot()
            ax_fg.set_xlabel('gt')


            #AHORA HACEMOS EL LOOP ENTRE LOS ARCHIVOS DE DIFERENTES PARAMETROS Y LOS PONEMOS EN SU CORRESPONDIENTE GRAFICO Y EJE
            for i,d in enumerate(d):
                g_str=str(g).replace('.','_')
                k_str=str(k).replace('.','_')
                J_str=str(J).replace('.','_')
                d_str=str(d).replace('.','_')
                x_str=str(x).replace('.','_')
                gamma_str=str(gamma).replace('.','_')
                p_str=str(p).replace('.','_')
                
                param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
                csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
                
                data=pd.read_csv(csvname,header=0)

                '''----DATOS DE LOS PLOTS----'''

                '''--- N=0 ---'''
                line0,=ax0.plot(g*data['t'], data['pr(gg0)'], color=colors1[i],label='gg0, d='+str(d))
                # ax0.legend([line0],[data.keys()[0]+', d='+str(d)])
                ax0.set_title(param_name)

                # plot_coherencias(data,9,ax0)#,0) #N=0

        
                '''--- N=1 ---'''
                line11,=ax1.plot(g*data['t'],data['pr(gg1)'],color=colors3[i],label=',d='+str(d))
                line12,=ax1.plot(g*data['t'],data['pr(eg0+ge0)'],color=colors3[i+len_d],label=',d='+str(d))
                line13,=ax1.plot(g*data['t'],data['pr(eg0-ge0)'],color=colors3[i+2*len_d],label=',d='+str(d))
                # plot_coherencias(data,3,ax1) #N=1
                # plot_coherencias(data,6,ax1) #N=1
                # plot_coherencias(data,10,ax1) #N=1
                ax1.set_title(param_name)
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*data['t'],data['pr(gg2)'],color=colors4[i],label=',d='+str(d))
                line22,=ax2.plot(g*data['t'],data['pr(eg1+ge1)'],color=colors4[i+len_d],label=',d='+str(d))
                line23,=ax2.plot(g*data['t'],data['pr(eg1-ge1)'],color=colors4[i+2*len_d],label=',d='+str(d))
                line24,=ax2.plot(g*data['t'],data['pr(ee0)'],color=colors4[i+3*len_d],label=',d='+str(d))
                # plot_coherencias(data,0,ax2) #N=2
                # plot_coherencias(data,4,ax2) #N=2
                # plot_coherencias(data,7,ax2) #N=2 
                # plot_coherencias(data,11,ax2) #N=2
                ax2.set_title(param_name)
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')
                '''----EVALS----'''
                for j in range(12): 
                    ax_eval.plot(g*data['t'],data['Eigenvalue '+str(j)],color=colors_eval[i][j],label=f"$\lambda_{j}$")
                ax_eval.legend()
                '''----FG-----'''

                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*data['t'],data['1/2 <sz1+sz2>'],color=colors3[i],label=',d='+str(d))
                line_p1,=ax_pauli.plot(g*data['t'],data['<sx1>'],color=colors3[i+len_d],label=',d='+str(d))
                line_p2,=ax_pauli.plot(g*data['t'],data['<sx2>'],color=colors3[i+2*len_d],label=',d='+str(d))
                ax_pauli.set_title(param_name)
                pauli_lines.append([line_p0,line_p1,line_p2])
                pauli_names.append(['$\\frac{1}{2}<\\sigma_z^{(1)}+\\sigma_z^{(2)}>$'+', d='+str(d),'$<\\sigma_x^{(1)}>$'+', d='+str(d),'$<\\sigma_x^{(2)}>$'+', d='+str(d)])
            
                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*data['t'],data['S von Neuman tot'],color=colors2[i],label='d='+str(d))
                lineSlin,=ax_Slin.plot(g*data['t'],data['S lineal tot'],color=colors2[i+len_d],label='d='+str(d))
                ax_Svn.set_title(param_name)
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*data['t'],data['S vN atom'],color=colors3[i],label='d='+str(d))
                lineSrlin,=ax_Srlin.plot(g*data['t'],data['S lin atom'],color=colors3[i+len_d],label='d='+str(d))
                lineCon,=ax_Con.plot(g*data['t'],data['Concu atom'],color=colors3[i+2*len_d],label='d='+str(d))
                ax_Srvn.set_title(param_name)
                # ax_Srvn.legend([lineSrvn,lineSrlin,lineCon],['S_vN'+', d='+str(d),'S_lin'+', d='+str(d),'Conc'+', d='+str(d)])
        
        ax_pauli.legend()#[np.array(pauli_lines).flatten()],[np.array(pauli_names).flatten()])
        ax_Svn.legend()#[lineSvn,lineSlin],['S_vN'+', d='+str(d),'S_lin'+', d='+str(d)])

        if savePlots==True:
            script_path=os.path.dirname(__file__)            
            relative_path="graficos resumen"+"\\"+ci+"\\"+"delta"
            path=os.path.join(script_path, relative_path)
            if os.path.exists(path):
                os.chdir(path)
            else: 
                os.makedirs(path)
                os.chdir(path)

            fig0.savefig(ci+' n=0 '+folder_names,dpi=100)
            fig2.savefig(ci+' n=2 '+folder_names,dpi=100)
            fig1.savefig(ci+' n=1 '+folder_names,dpi=100)
            fig_pauli.savefig(ci+' pauli '+folder_names,dpi=100)
            fig_S.savefig(ci+' entropia '+folder_names,dpi=100)
            fig_Sr.savefig(ci+' entropia reducida '+folder_names,dpi=100)
            fig_autoval.savefig()
            fig_fg.savefig()
            plt.close()

        elif showPlots==True:
            plt.show()
        else:
            print("Ni savePlots ni showPlots es True...")

def plot_cis(cis:list,x:float,d:float,gamma:float,modelo:int=0,savePlots:bool=False):
    """
    Params:
    -cis: lista de strings con nombres de condiciones iniciales
    -modelo: [0:\'disipativo lineal\', 1:\'disipativo bs\', 2:\'unitario lineal\', 3:\'unitario bs\']"""
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
    folder_name=["disipativo lineal","disipativo bs","unitario lineal","unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
    if modelo==0:
        folder_name=folder_name[0]
    elif modelo==1:
        folder_name=folder_name[1]
    elif modelo==2:
        folder_name=folder_name[2]
    elif modelo==3:
        folder_name=folder_name[3]
    else:
        print("Param \'modelo\' tiene que ser 0:\'disipativo lineal\', 1:\'disipativo bs\', 2:\'unitario lineal\', 3:\'unitario bs\'")


    cmap1=mpl.colormaps["plasma"]
    colors1=cmap1(np.linspace(0,1,len(cis)))
    colors2=cmap1(np.linspace(0,1,2*len(cis)))
    colors3=cmap1(np.linspace(0,1,3*len(cis)))
    colors4=cmap1(np.linspace(0,1,4*len(cis)))
    colors_eval=[mpl.colormaps["Purples"](np.linspace(0,1,12)),mpl.colormaps["Blues"](np.linspace(0,1,12)),mpl.colormaps["Greens"](np.linspace(0,1,12)),mpl.colormaps["Oranges"](np.linspace(0,1,12))]


    '''-------LAYOUT PARA LOS GRAFICOS------'''
    #PARA CADA GRAFICO QUE VAMOS A HACER, CREAMOS LA FIGURA EN UNA PRIMERA INSTANCIA ASI QUEDAN ESTATICOS, Y DESPUES HACEMOS UN LOOP POR LOS ARCHIVOS QUE VAN A ESTAR
    #INCLUIDOS EN CADA UNO PARA HACER LA COMPARACION
    '''N=0'''
    fig0 = plt.figure(figsize=(16,9))
    fig0.suptitle('N=0 '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax0 = fig0.add_subplot()
    ax0.set_xlabel('gt')
    ax0.set_ylabel('Amp. Prob. ')
    ax0.set_ylim(0,1)
    '''N=1'''
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot()
    fig1.suptitle('N=1 '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax1.set_xlabel('gt')
    ax1.set_ylabel('Amp. Prob. ')
    ax1.set_ylim(0,1)
    '''N=2'''
    fig2 = plt.figure(figsize=(16,9))
    ax2 = fig2.add_subplot()
    fig2.suptitle('N=2 '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax2.set_xlabel('gt')
    ax2.set_ylabel('Amp. Prob. ')
    # ax2.set_zlim(0,1)

    '''PAULI'''
    fig_pauli = plt.figure(figsize=(16,9))
    ax_pauli = fig_pauli.add_subplot()
    fig_pauli.suptitle('Pauli '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_pauli.set_xlabel('gt')
    ax_pauli.set_ylabel('V.M.')
    ax_pauli.set_ylim(-1,1)
    '''ENTROPIA VON NEUMAN Y LINEAL'''
    fig_S = plt.figure(figsize=(16,9))
    ax_Slin = fig_S.add_subplot(121)
    ax_Svn = fig_S.add_subplot(122)
    fig_S.suptitle('Entropia A-A-F '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_Svn.set_ylabel('S')
    ax_Svn.set_title('$S_{vn}$')
    ax_Slin.set_ylabel('S')
    ax_Slin.set_title('$S_{lin}$')
    ax_Slin.set_xlabel('gt')
    ax_Svn.set_xlabel('gt')
    ax_Svn.set_ylim(0,np.log(8))
    ax_Slin.set_ylim(0,1)

    '''ESTADO REDUCIDO: ENTROPIA Y CONCURRENCIA'''
    fig_Sr = plt.figure(figsize=(16,9))
    ax_Srlin = fig_Sr.add_subplot(131)
    ax_Srvn = fig_Sr.add_subplot(132)
    ax_Con = fig_Sr.add_subplot(133)
    fig_Sr.suptitle('Entropia Reducida '+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_Srvn.set_ylabel('S')
    ax_Srvn.set_title('$S_{vN}$')
    ax_Srlin.set_ylabel('S')
    ax_Srlin.set_title('$S_{lin}$')
    ax_Con.set_ylabel('C')
    ax_Con.set_title('Concurrence')
    ax_Con.set_xlabel('gt')
    ax_Srlin.set_xlabel('gt')
    ax_Srvn.set_xlabel('gt')
    ax_Srvn.set_ylim(0,np.log(8))
    ax_Srlin.set_ylim(0,1)
    ax_Con.set_ylim(0,1)

    '''----Autovalores----'''
    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()
    ax_eval.set_xlabel('gt')
    ax_eval.set_ylabel('Eval')
    
    '''-----FG------'''
    fig_fg=plt.figure()
    fig_fg.suptitle("Fase Geometrica "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_fg=fig_fg.add_subplot()
    ax_fg.set_xlabel('gt')
    
    for i,ci in enumerate(cis):
        relative_path="datos"+"\\"+folder_name+"\\"+ci 
        path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
        if os.path.exists(path):
            os.chdir(path)
        else: 
            print("Dir %s does not exist", path)
        g_str=str(g).replace('.','_')
        k_str=str(k).replace('.','_')
        J_str=str(J).replace('.','_')
        d_str=str(d).replace('.','_')
        x_str=str(x).replace('.','_')
        gamma_str=str(gamma).replace('.','_')
        p_str=str(p).replace('.','_')
        param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
        csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
        
        data=pd.read_csv(csvname,header=0)

        '''----DATOS DE LOS PLOTS----'''

        '''--- N=0 ---'''
        line0,=ax0.plot(g*data['t'], data['pr(gg0)'], color=colors1[i],label='gg0,'+ci)
        ax0.set_title(param_name)
        # plot_coherencias(data,9,ax0)#,0) #N=0

        '''--- N=1 ---'''
        line11,=ax1.plot(g*data['t'],data['pr(gg1)'],color=colors3[i],label='gg1,'+ci)
        line12,=ax1.plot(g*data['t'],data['pr(eg0+ge0)'],color=colors3[i+len(cis)],label='eg0+,'+ci)
        line13,=ax1.plot(g*data['t'],data['pr(eg0-ge0)'],color=colors3[i+2*len(cis)],label='eg0-,'+ci)
        # plot_coherencias(data,3,ax1) #N=1
        # plot_coherencias(data,6,ax1) #N=1
        # plot_coherencias(data,10,ax1) #N=1
        ax1.set_title(param_name)
        
        '''--- N=2 ---'''
        line21,=ax2.plot(g*data['t'],data['pr(gg2)'],color=colors4[i],label='gg2,'+ci)
        line22,=ax2.plot(g*data['t'],data['pr(eg1+ge1)'],color=colors4[i+len(cis)],label='eg1+,'+ci)
        line23,=ax2.plot(g*data['t'],data['pr(eg1-ge1)'],color=colors4[i+2*len(cis)],label='eg1-,'+ci)
        line24,=ax2.plot(g*data['t'],data['pr(ee0)'],color=colors4[i+3*len(cis)],label='ee0,'+ci)
        # plot_coherencias(data,0,ax2) #N=2
        # plot_coherencias(data,4,ax2) #N=2
        # plot_coherencias(data,7,ax2) #N=2 
        # plot_coherencias(data,11,ax2) #N=2
        ax2.set_title(param_name)

        # '''--- N=3 ---'''

        # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
        # ax=[ax]
        # fig.suptitle('N=3')
        # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
        # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
        # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')
        '''----EVALS----'''
        for j in range(12): 
            ax_eval.plot(g*data['t'],data['Eigenvalue '+str(j)],color=colors_eval[i][j])
        # ax_eval.legend()
        '''----FG-----'''
        ax_fg.plot(g*data['t'],data['FG'],color=colors1[i],label=f'$|\psi_0>=|{ci}>')
        '''--- VM Pauli ---'''

        line_p0,=ax_pauli.plot(g*data['t'],data['1/2 <sz1+sz2>'],color=colors3[i],label='$\\frac{1}{2}<\\sigma_z^{(1)}+\\sigma_z^{(2)}>$'+','+ci)
        line_p1,=ax_pauli.plot(g*data['t'],data['<sx1>'],color=colors3[i+len(cis)],label='$<\\sigma_x^{(1)}>$'+','+ci)
        line_p2,=ax_pauli.plot(g*data['t'],data['<sx2>'],color=colors3[i+2*len(cis)],label='$<\\sigma_x^{(2)}>$'+','+ci)
        ax_pauli.set_title(param_name)
        
        '''--- Entropias ---'''
        #PLOT PARA LAS ENTROPIAS
        
        lineSvn,=ax_Svn.plot(g*data['t'],data['S von Neuman tot'],color=colors1[i], label=ci)
        lineSlin,=ax_Slin.plot(g*data['t'],data['S lineal tot'],color=colors1[i], label=ci)
        ax_Svn.set_title(param_name)
        #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

        '''---Trazamos sobre el campo---'''
        #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
        #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

        lineSrvn,=ax_Srvn.plot(g*data['t'],data['S vN atom'],color=colors1[i], label=ci)
        lineSrlin,=ax_Srlin.plot(g*data['t'],data['S lin atom'],color=colors1[i], label=ci)
        lineCon,=ax_Con.plot(g*data['t'],data['Concu atom'],color=colors1[i], label=ci)
        ax_Srvn.set_title(param_name)

    ax0.legend()
    ax1.legend()    
    ax2.legend()    
    ax_Slin.legend()
    ax_Srlin.legend()
    ax_Con.legend()
    ax_pauli.legend()#pauli_lines,pauli_names)
    ax_Slin.legend()#sLinLines,sLinNames)
    ax_Svn.legend()#sVNLines,sVNNames)
    ax_fg.legend()


    if savePlots==True:
        script_path=os.path.dirname(__file__)            
        relative_path="graficos"+"\\comparacion"
        path=os.path.join(script_path, relative_path)
        if os.path.exists(path):
            os.chdir(path)
        else: 
            os.makedirs(path)
            os.chdir(path)

        
        # fig0.savefig(f'comparacion {cis} n=0 '+folder_name,dpi=100)
        # fig2.savefig(f'comparacion {cis} n=2 '+folder_name,dpi=100)
        # fig1.savefig(f'comparacion {cis} n=1 '+folder_name,dpi=100)
        # fig_pauli.savefig(f'comparacion {cis} pauli '+folder_name,dpi=100)
        # fig_S.savefig(f'comparacion {cis} entropia '+folder_name,dpi=100)
        # fig_Sr.savefig(f'comparacion {cis} entropia reducida '+folder_name,dpi=100)
        # fig_autoval.savefig(f'comparacion {cis} autovals '+folder_name,dpi=100)
        fig_fg.savefig(f'comparacion {cis} FG d={str(d/g).replace('.','_')}g x={str(x/g).replace('.','_')}g '+folder_name,dpi=100)
        plt.close()

    else:
        plt.show()

def plot_fg_delta(ci:str,delta:list,x:float,folder_name:str="unitario lineal"):
    gamma=0.1*g
    script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU    
    cmap=mpl.colormaps["viridis"]
    colors=cmap(np.linspace(0,1,len(delta)))
    norm = mcolors.Normalize(vmin=delta[0]/g, vmax=delta[-1]/g)
    
    fig_n0=plt.figure()
    fig_n0.suptitle("N=0 "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_n0=fig_n0.add_subplot()
    ax_n0.set_xlabel('gt')
    fig_n0.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax_n0,orientation='vertical',label='$\Delta$')
    
    fig_n1=plt.figure()
    fig_n1.suptitle("N=1 "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_n1_1=fig_n1.add_subplot(131,)
    ax_n1_1.set_xlabel('gt')
    ax_n1_1.set_title('gg1')
    ax_n1_2=fig_n1.add_subplot(132)
    ax_n1_2.set_xlabel('gt')
    ax_n1_2.set_title('eg0+ge0')
    ax_n1_3=fig_n1.add_subplot(133)
    ax_n1_3.set_xlabel('gt')
    ax_n1_3.set_title('eg0-ge0')

    fig_n1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax_n1_3,orientation='vertical',label='$\Delta$')

    fig_fg=plt.figure()
    fig_fg.suptitle("Fase Geometrica "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax_fg=fig_fg.add_subplot()
    ax_fg.set_xlabel('gt')
    ax_fg.set_title(ci)
    fig_fg.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax_fg,orientation='vertical',label='$\Delta$')
    for i,d in enumerate(delta):
        relative_path="datos"+"\\"+folder_name+"\\"+ci 
        path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
        if os.path.exists(path):
            os.chdir(path)
        else: 
            print("Dir %s does not exist", path)
        g_str=str(g).replace('.','_')
        k_str=str(k).replace('.','_')
        J_str=str(J).replace('.','_')
        d_str=str(d).replace('.','_')
        x_str=str(x).replace('.','_')
        gamma_str=str(gamma).replace('.','_')
        p_str=str(p).replace('.','_')
        
        param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
        csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
        
        data=pd.read_csv(csvname,header=0)

        ax_n0.plot(g*data['t'],data['pr(gg0)'],color=colors[i])

        ax_n1_1.plot(g*data['t'],data['pr(gg1)'],color=colors[i])
        ax_n1_2.plot(g*data['t'],data['pr(eg0+ge0)'],color=colors[i])
        ax_n1_3.plot(g*data['t'],data['pr(eg0-ge0)'],color=colors[i])

        ax_fg.plot(g*data['t'],data['FG'],color=colors[i])

    plt.show()

def anim(ci:str,folder_name:str,key:list,delta:list,x:float,gamma:float):
    mpl.use('TkAgg')
    relative_path="datos"+"\\"+folder_name+"\\"+ci 
    path=os.path.join(script_path, relative_path) #CAMBIAMOS EL CHDIR A LA CARPETA DONDE QUEREMOS BUSCAR LOS ARCHIVOS
    if os.path.exists(path):
        os.chdir(path)
    else: 
        print("Dir %s does not exist", path)
    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')

        # Create the figure and axes
    fig= plt.figure()
    fig.suptitle("$|\psi_0>=$"+ci+" ; "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1])
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)


    # Initialize a plot object (e.g., a line plot)
    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)
    line3, = ax3.plot([], [], lw=2)

    # Create a colormap and normalize it to the number of frames
    cmap = mpl.colormaps['viridis']   # Viridis colormap with as many colors as CSV files
    norm = mcolors.Normalize(vmin=delta[0]/g, vmax=delta[-1]/g)

    # Add the colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable requires an array, but we don't need it
    cbar = plt.colorbar(sm, ax=ax3, orientation='vertical')
    cbar.set_label('$\Delta/g$')

    # Create a black rectangle to indicate the current position in the colorbar
    rect_height = 1 / len(delta)  # Height of the rectangle
    current_color_rect = Rectangle((0, 0), width=1, height=rect_height, color='black', lw=2, transform=cbar.ax.transAxes)
    cbar.ax.add_patch(current_color_rect)  # Add the rectangle to the colorbar axes

    # Set up the axis limits and labels
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    ax3.set_ylim(-1.5, 0.1)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    ax3.set_xlim(0, 20)
    ax1.set_xlabel('gt')
    ax2.set_xlabel('gt')
    ax3.set_xlabel('gt')
    ax1.set_title(key[0])
    ax2.set_title(key[1])
    ax3.set_title(key[2])

    # Define the initialization function
    def init():
        """Initialize the plot with empty data."""
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1,line2,line3

    # Define the update function for each frame
    def update(frame):
        """Read the CSV data and update the plot."""
        # Read the CSV file for the current frame
        d_str=str(delta[frame]).replace('.','_')
        csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
        data=pd.read_csv(csvname,header=0)

        # Update the plot data
        line1.set_data(g*data['t'], data[key[0]])
        line2.set_data(g*data['t'], data[key[1]])
        line3.set_data(g*data['t'], data[key[2]])
        # Update the line color based on the current frame
        color = cmap(norm(delta[0]/g+frame*(delta[-1]/g-delta[0]/g)/(len(delta) - 1)))
        line1.set_color(color)
        line2.set_color(color)
        line3.set_color(color)

        # Move the rectangle to the current position in the colorbar (keep it black)
        current_color_rect.set_y(frame / len(delta))  # Adjust y based on current frame
        return line1,line2,line3, current_color_rect

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=len(delta), init_func=init, repeat=True)

    # Show the plot
    plt.show()

    ani.save(script_path+"\\"+"gifs"+"\\"+"animation "+ci+" "+folder_name.split(" ")[-2]+" "+folder_name.split(" ")[-1]+".gif", writer='pillow')


#CONDICIONES INICIALES EN FOLDER condiciones_iniciales=["ee0","eg0","gg1","eg0-","eg1-","eg1+ge0","gg2","w1","eg1-ge0"] 
#PARECE NO FUNCIONAR CON MAS DE UNA CONDICION INICIAL A LA VEZ
# for x in [0,1/4*g,1/2*g]:
#     for d in [0.5*g,g,2*g]:
#         plot_cis(['w'],x=x,d=d,gamma=0.1*g,savePlots=True)
# plot3D_delta(['eg0+'])#,'eg0-','eg0+'])

# plot_fg_delta('w',delta=[0,0.2*g,0.5*g,0.9*g,g,1.2*g,1.5*g,1.6*g,2*g],x=0,folder_name="10_3_8 unitario lineal")
delta=[0,0.1*g,0.2*g,0.3*g,0.4*g,0.5*g,0.6*g,0.7*g,0.8*g,0.9*g,g,1.1*g,1.2*g,1.3*g,1.4*g,1.5*g,1.6*g,1.7*g,1.8*g,1.9*g,2*g]
anim('w',"10_3_8 unitario bs",['pr(gg1)','pr(eg0+ge0)','FG'],delta,0,0.1*g)


