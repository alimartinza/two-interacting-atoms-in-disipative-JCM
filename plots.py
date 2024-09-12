from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
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
t_final=100000
steps=100000
t=np.linspace(0,t_final,steps)
save_plot=False
plot_show=True

def plot_ReIm_coherencias(n:int,ax,xlabel=None,ylabel=None):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    colors = plt.cm.jet(np.linspace(0,1,12))
    i=0
    for key in coherencias.keys():
        if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                ax.plot(g*t,np.real(coherencias[key]),linestyle='dashed',label=f'Re[C({key})]',color=colors[i])
                ax.plot(g*t,np.imag(coherencias[key]),linestyle='dashdot',label=f'Im[C({key})]',color=colors[i])
                i+=1
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_coherencias(n:int,ax,xlabel='gt',ylabel='Abs(Coh)'):
    '''
    Parametros
    - n: numero del vector de la base del cual se quieren graficar las coherencias
    -n_ax: en que ax queres graficar todas las coherencias
    
    Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
    #colors = ['#000000','#000000','#000000','#ff7043','#000000','#000000','#000000','#000000','#000000','#1976d2','#4caf50','#000000'] 
    colors=plt.cm.jet(np.linspace(0,1,12))
    i=0
    if n==1:
        for key in ['0,1','1,2','1,3','1,4','1,5','1,6','1,7','1,8','1,9','1,10','1,11']:
            ax.plot(g*t,np.abs(data[key]),linestyle='dashed',color=colors[i])#,label=f'C({key})')
            i+=1
    else:
        for key in data.keys():
            if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                    ax.plot(g*t,np.abs(data[key]),linestyle='dashed',color=colors[i])#,label=f'C({key})')
                    i+=1
    # ax.legend()
    # # ax[n_ax].set_xlabel(xlabel)
    # ax[n_ax].set_ylabel(ylabel)

def plot_gamma(condiciones_iniciales:list):
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

                '''----DATOS DE LOS PLOTS----'''
                '''CHECK TR=1'''
                trace=data['pr(gg0)']+data['pr(gg1)']+data['pr(gg2)']+data['pr(ee0)']+data['pr(eg0+ge0)']+data['pr(ge0-eg0)']+data['pr(eg1+ge1)']+data['pr(eg1-ge1)']
                ax_check.plot(g*t,trace,color=colors[i], alpha=0.8)
                '''--- N=0 ---'''
                line0,=ax0.plot(g*t, data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],[data.keys()[0]])
                # plot_coherencias(9,ax0)#,0) #N=0
                # # plot_coherencias(3,ax11) #N=1
                # # plot_coherencias(4,ax12) #N=1
                # # plot_coherencias(10,ax13) #N=1
                # plot_coherencias(0,0) #N=2
                # plot_coherencias(5,1) #N=2
                # plot_coherencias(6,2) #N=2 ESTA TIENE ALGUN PROBLEMA, SE GRAFICAN EL C(6,9) Y c(6,3) (CREO QUE ESOS) PERO DEBERIAN SER 0, Y SE GRAFICAN MUCHO NO ES ERROR NUMERICO
                # plot_coherencias(11,3) #N=2
                # plot_coherencias(1,0) #N=3
                # plot_coherencias(7,0) #N=3
                # plot_coherencias(8,0) #N=3

                
                '''--- N=1 ---'''
                line11,=ax1.plot(g*t,data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line12,=ax1.plot(g*t,data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*t,data['pr(ge0-eg0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*t,data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*t,data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*t,data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*t,data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*t,data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*t,data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*t,data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*t,data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*t,data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*t,data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*t,data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*t,data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

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

def plot_x(condiciones_iniciales:list):
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
                line0,=ax0.plot(g*t, data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],[data.keys()[0]])
                # plot_coherencias(9,ax0)#,0) #N=0
                # # plot_coherencias(3,ax11) #N=1
                # # plot_coherencias(4,ax12) #N=1
                # # plot_coherencias(10,ax13) #N=1
                # plot_coherencias(0,0) #N=2
                # plot_coherencias(5,1) #N=2
                # plot_coherencias(6,2) #N=2 ESTA TIENE ALGUN PROBLEMA, SE GRAFICAN EL C(6,9) Y c(6,3) (CREO QUE ESOS) PERO DEBERIAN SER 0, Y SE GRAFICAN MUCHO NO ES ERROR NUMERICO
                # plot_coherencias(11,3) #N=2
                # plot_coherencias(1,0) #N=3
                # plot_coherencias(7,0) #N=3
                # plot_coherencias(8,0) #N=3

                
                '''--- N=1 ---'''
                line11,=ax1.plot(g*t,data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line12,=ax1.plot(g*t,data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*t,data['pr(ge0-eg0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*t,data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*t,data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*t,data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*t,data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*t,data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*t,data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*t,data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*t,data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*t,data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*t,data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*t,data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*t,data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

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

def plot_delta(condiciones_iniciales:list):
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
            x=1/4*g#[0,1/4*g,1/2*g]
            d=[0,0.5*g,2*g]
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
            ax0.set_yticks([0,1,2],np.array(d)/g)
            ax0.set_zlim(0,1)
            '''N=1'''
            fig1 = plt.figure(figsize=(16,9))
            ax1 = fig1.add_subplot(projection='3d')
            fig1.suptitle('N=1 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax1.set_xlabel('gt')
            ax1.set_ylabel('$\\Delta$/g')
            ax1.set_zlabel('Amp. Prob. ')
            ax1.set_yticks([0,1,2],np.array(d)/g)
            ax1.view_init(30,-40,0)
            ax1.set_yticks([0,1,2],np.array(d)/g)
            ax1.set_zlim(0,1)

            '''N=2'''
            fig2 = plt.figure(figsize=(16,9))
            ax2 = fig2.add_subplot(projection='3d')
            fig2.suptitle('N=2 '+folder_names.split(" ")[-2]+" "+folder_names.split(" ")[-1])
            ax2.set_xlabel('gt')
            ax2.set_ylabel('$\\Delta$/g')
            ax2.set_zlabel('Amp. Prob. ')
            ax2.set_yticks([0,1,2],np.array(d)/g)
            ax2.view_init(30,-40,0)
            ax2.set_yticks([0,1,2],np.array(d)/g)
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
            ax_pauli.set_yticks([0,1,2],np.array(d)/g)
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
            ax_Svn.set_yticks([0,1,2],np.array(d)/g)
            ax_Slin.set_yticks([0,1,2],np.array(d)/g)
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
            ax_Srvn.set_yticks([0,1,2],np.array(d)/g)
            ax_Srlin.set_yticks([0,1,2],np.array(d)/g)
            ax_Con.set_yticks([0,1,2],np.array(d)/g)
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
                
                data=pd.read_csv(csvname,header=0,index_col=0)

                '''----DATOS DE LOS PLOTS----'''

                '''--- N=0 ---'''
                line0,=ax0.plot(g*t, data['pr(gg0)'], zs=i, zdir='y', color=colors[0], alpha=0.8)
                ax0.legend([line0],[data.keys()[0]])
                # plot_coherencias(9,ax0)#,0) #N=0
                # # plot_coherencias(3,ax11) #N=1
                # # plot_coherencias(4,ax12) #N=1
                # # plot_coherencias(10,ax13) #N=1
                # plot_coherencias(0,0) #N=2
                # plot_coherencias(5,1) #N=2
                # plot_coherencias(6,2) #N=2 ESTA TIENE ALGUN PROBLEMA, SE GRAFICAN EL C(6,9) Y c(6,3) (CREO QUE ESOS) PERO DEBERIAN SER 0, Y SE GRAFICAN MUCHO NO ES ERROR NUMERICO
                # plot_coherencias(11,3) #N=2
                # plot_coherencias(1,0) #N=3
                # plot_coherencias(7,0) #N=3
                # plot_coherencias(8,0) #N=3

                
                '''--- N=1 ---'''
                line11,=ax1.plot(g*t,data['pr(gg1)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line12,=ax1.plot(g*t,data['pr(eg0+ge0)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line13,=ax1.plot(g*t,data['pr(ge0-eg0)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                ax1.legend([line11,line12,line13],[data.keys()[1],data.keys()[2],data.keys()[3]])
                
                '''--- N=2 ---'''

                line21,=ax2.plot(g*t,data['pr(gg2)'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line22,=ax2.plot(g*t,data['pr(eg1+ge1)'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line23,=ax2.plot(g*t,data['pr(eg1-ge1)'],zs=i, zdir='y', color=colors[2], alpha=0.8)
                line24,=ax2.plot(g*t,data['pr(ee0)'],zs=i, zdir='y', color=colors[3], alpha=0.8)
                ax2.legend([line21,line22,line23,line24],[data.keys()[4],data.keys()[5],data.keys()[6],data.keys()[7]])
                # '''--- N=3 ---'''

                # fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
                # ax=[ax]
                # fig.suptitle('N=3')
                # ax[0].plot(g*t,data['pr(eg2)'],label=data.keys()[8],color='black')
                # ax[0].plot(g*t,data['pr(ge2)'],label=data.keys()[9],color='blue')
                # ax[0].plot(g*t,data['pr(ee1)'],label=data.keys()[10],color='red')

    
                '''--- VM Pauli ---'''

                line_p0,=ax_pauli.plot(g*t,data['1/2 <sz1+sz2>'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                line_p1,=ax_pauli.plot(g*t,data['<sx1>'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                line_p2,=ax_pauli.plot(g*t,data['<sx2>'],zs=i, zdir='y', color=colors[2], alpha=0.8)

                ax_pauli.legend([line_p0,line_p1,line_p2],['$\\frac{1}{2}<\\sigma_z^{(1)}+\\sigma_z^{(2)}>$','$<\\sigma_x^{(1)}>$','$<\\sigma_x^{(2)}>$'])#[data.keys()[11],data.keys()[12],data.keys()[13]])

                '''--- Entropias ---'''
                #PLOT PARA LAS ENTROPIAS
                
                lineSvn,=ax_Svn.plot(g*t,data['S von Neuman tot'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSlin,=ax_Slin.plot(g*t,data['S lineal tot'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                ax_Svn.legend([lineSvn,lineSlin],['S_vN','S_lin'])
                #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA ANIMACION ESTARIA COPADO

                '''---Trazamos sobre el campo---'''
                #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS
                #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES

                lineSrvn,=ax_Srvn.plot(g*t,data['S vN atom'],zs=i, zdir='y', color=colors[0], alpha=0.8)
                lineSrlin,=ax_Srlin.plot(g*t,data['S lin atom'],zs=i, zdir='y', color=colors[1], alpha=0.8)
                lineCon,=ax_Con.plot(g*t,data['Concu atom'],zs=i, zdir='y', color=colors[2], alpha=0.8)

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
#CONDICIONES INICIALES EN FOLDER condiciones_iniciales=["ee0","eg0","gg1","eg0-","eg1-","eg1+ge0","gg2","w1","eg1-ge0"] 
#PARECE NO FUNCIONAR CON MAS DE UNA CONDICION INICIAL A LA VEZ
plot_gamma(['w1'])
plot_delta(["w1"])
plot_x(["w1"])
