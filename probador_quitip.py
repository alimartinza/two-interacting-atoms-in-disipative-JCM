from qutip import *
import numpy as np

psi0=tensor(basis(2,0),basis(2,1))
rho0=ket2dm(psi0)


e=basis(2,0)
gr=basis(2,1)

ee0=tensor(e,e,basis(3,0))
ee1=tensor(e,e,basis(3,1))
ee2=tensor(e,e,basis(3,2))

eg0=tensor(e,gr,basis(3,0))
ge0=tensor(gr,e,basis(3,0))

eg1=tensor(e,gr,basis(3,1))
ge1=tensor(gr,e,basis(3,1))

eg2=tensor(e,gr,basis(3,2))
ge2=tensor(gr,e,basis(3,2))

gg0=tensor(gr,gr,basis(3,0))
gg1=tensor(gr,gr,basis(3,1))
gg2=tensor(gr,gr,basis(3,2))

# print(eg1+2*eg2+3*ge0+4*ge1+5*ge2+6*gg0+7*gg1+8*gg2)

vals=ket2dm(gg0).eigenenergies()
nzvals=vals[vals != 0]
print(nzvals)

m=np.arange(16).reshape([4,4])
print(m)
print(m[1][0])
for i in range(13,12):
    print(i)

n=10
coherencias={'0,10':[],'10,10':[]}
for key in coherencias.keys():
    if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
        print('sape loquita')
        

'''-------------------CODIGO SUPER LARGO PARA PLOTEAR LAS COHERENCIAS PERO FUNCIONA BIEN----------------------'''
'''
colors = plt.cm.jet(np.linspace(0,1,len(nomb[0])))
ax[0].plot(sol.times,sol.expect[0],label=nomb[0][0],color='black')
ax[0].plot(sol.times,np.real(coherencias['0,9']),linestyle='dashed',label='Re{C(0,9)}',color='blue')
ax[0].plot(sol.times,np.real(coherencias['1,9']),linestyle='dashed',label='Re{C(1,9)}',color='blue',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['2,9']),linestyle='dashed',label='Re{C(2,9)}',color='green')
ax[0].plot(sol.times,np.real(coherencias['3,9']),linestyle='dashed',label='Re{C(3,9)}',color='green',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['4,9']),linestyle='dashed',label='Re{C(4,9)}',color='pink')
ax[0].plot(sol.times,np.real(coherencias['5,9']),linestyle='dashed',label='Re{C(5,9)}',color='pink',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['6,9']),linestyle='dashed',label='Re{C(6,9)}',color='red')
ax[0].plot(sol.times,np.real(coherencias['7,9']),linestyle='dashed',label='Re{C(7,9)}',color='red',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['8,9']),linestyle='dashed',label='Re{C(8,9)}',color='cyan')
ax[0].plot(sol.times,np.real(coherencias['9,10']),linestyle='dashed',label='Re{C(9,10)}',color='cyan',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['9,11']),linestyle='dashed',label='Re{C(9,11)}',color='magenta')

ax[0].plot(sol.times,np.imag(coherencias['0,9']),linestyle='dashdot',label='Im{C(0,9)}',color='blue')
ax[0].plot(sol.times,np.imag(coherencias['1,9']),linestyle='dashdot',label='Im{C(1,9)}',color='blue',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['2,9']),linestyle='dashdot',label='Im{C(2,9)}',color='green')
ax[0].plot(sol.times,np.imag(coherencias['3,9']),linestyle='dashdot',label='Im{C(3,9)}',color='green',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['4,9']),linestyle='dashdot',label='Im{C(4,9)}',color='pink')
ax[0].plot(sol.times,np.imag(coherencias['5,9']),linestyle='dashdot',label='Im{C(5,9)}',color='pink',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['6,9']),linestyle='dashdot',label='Im{C(6,9)}',color='red')
ax[0].plot(sol.times,np.imag(coherencias['7,9']),linestyle='dashdot',label='Im{C(7,9)}',color='red',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['8,9']),linestyle='dashdot',label='Im{C(8,9)}',color='cyan')
ax[0].plot(sol.times,np.imag(coherencias['9,10']),linestyle='dashdot',label='Im{C(9,10)}',color='cyan',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['9,11']),linestyle='dashdot',label='Im{C(9,11)}',color='magenta')

ax[0].set_xlabel('t')
ax[0].set_ylabel(f'N=0 exitaciones')
ax[0].legend()

colors = plt.cm.jet(np.linspace(0,1,len(nomb[1])))
ax[1].plot(sol.times,sol.expect[1],label=nomb[1][0],color=colors[0])
ax[1].plot(sol.times,sol.expect[2],label=nomb[1][1],color=colors[1])
ax[1].plot(sol.times,sol.expect[3],label=nomb[1][2],color=colors[2])
ax[1].plot(sol.times,np.real(coherencias['0,10']),linestyle='dashed',label='Re{C(0,10)}',color='blue')
ax[1].plot(sol.times,np.real(coherencias['1,10']),linestyle='dashed',label='Re{C(1,10)}',color='blue',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['2,10']),linestyle='dashed',label='Re{C(2,10)}',color='green')
ax[1].plot(sol.times,np.real(coherencias['3,10']),linestyle='dashed',label='Re{C(3,10)}',color='green',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['4,10']),linestyle='dashed',label='Re{C(4,10)}',color='pink')
ax[1].plot(sol.times,np.real(coherencias['5,10']),linestyle='dashed',label='Re{C(5,10)}',color='pink',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['6,10']),linestyle='dashed',label='Re{C(6,10)}',color='red')
ax[1].plot(sol.times,np.real(coherencias['7,10']),linestyle='dashed',label='Re{C(7,10)}',color='red',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['8,10']),linestyle='dashed',label='Re{C(8,10)}',color='cyan')
ax[1].plot(sol.times,np.real(coherencias['9,10']),linestyle='dashed',label='Re{C(9,10)}',color='cyan',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['10,11']),linestyle='dashed',label='Re{C(10,11)}',color='magenta')

ax[1].plot(sol.times,np.imag(coherencias['0,10']),linestyle='dashdot',label='Im{C(0,10)}',color='blue')
ax[1].plot(sol.times,np.imag(coherencias['1,10']),linestyle='dashdot',label='Im{C(1,10)}',color='blue',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['2,10']),linestyle='dashdot',label='Im{C(2,10)}',color='green')
ax[1].plot(sol.times,np.imag(coherencias['3,10']),linestyle='dashdot',label='Im{C(3,10)}',color='green',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['4,10']),linestyle='dashdot',label='Im{C(4,10)}',color='pink')
ax[1].plot(sol.times,np.imag(coherencias['5,10']),linestyle='dashdot',label='Im{C(5,10)}',color='pink',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['6,10']),linestyle='dashdot',label='Im{C(6,10)}',color='red')
ax[1].plot(sol.times,np.imag(coherencias['7,10']),linestyle='dashdot',label='Im{C(7,10)}',color='red',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['8,10']),linestyle='dashdot',label='Im{C(8,10)}',color='cyan')
ax[1].plot(sol.times,np.imag(coherencias['9,10']),linestyle='dashdot',label='Im{C(9,10)}',color='cyan',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['10,11']),linestyle='dashdot',label='Im{C(10,11)}',color='magenta')


ax[2].plot(sol.times,sol.expect[1],label=nomb[1][0],color=colors[0])
ax[2].plot(sol.times,sol.expect[2],label=nomb[1][1],color=colors[1])
ax[2].plot(sol.times,sol.expect[3],label=nomb[1][2],color=colors[2])
ax[2].plot(sol.times,np.real(coherencias['0,6']),linestyle='dashed',label='Re{C(0,6)}',color='blue')
ax[2].plot(sol.times,np.real(coherencias['1,6']),linestyle='dashed',label='Re{C(1,6)}',color='blue',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['2,6']),linestyle='dashed',label='Re{C(2,6)}',color='green')
ax[2].plot(sol.times,np.real(coherencias['3,6']),linestyle='dashed',label='Re{C(3,6)}',color='green',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['4,6']),linestyle='dashed',label='Re{C(4,6)}',color='pink')
ax[2].plot(sol.times,np.real(coherencias['5,6']),linestyle='dashed',label='Re{C(5,6)}',color='pink',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['6,7']),linestyle='dashed',label='Re{C(6,7)}',color='red')
ax[2].plot(sol.times,np.real(coherencias['6,8']),linestyle='dashed',label='Re{C(6,8)}',color='red',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['6,9']),linestyle='dashed',label='Re{C(6,9)}',color='cyan')
ax[2].plot(sol.times,np.real(coherencias['6,10']),linestyle='dashed',label='Re{C(6,10)}',color='cyan',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['6,11']),linestyle='dashed',label='Re{C(6,11)}',color='magenta')

ax[2].plot(sol.times,np.imag(coherencias['0,6']),linestyle='dashdot',label='Im{C(0,6)}',color='blue')
ax[2].plot(sol.times,np.imag(coherencias['1,6']),linestyle='dashdot',label='Im{C(1,6)}',color='blue',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['2,6']),linestyle='dashdot',label='Im{C(2,6)}',color='green')
ax[2].plot(sol.times,np.imag(coherencias['3,6']),linestyle='dashdot',label='Im{C(3,6)}',color='green',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['4,6']),linestyle='dashdot',label='Im{C(4,6)}',color='pink')
ax[2].plot(sol.times,np.imag(coherencias['5,6']),linestyle='dashdot',label='Im{C(5,6)}',color='pink',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['6,7']),linestyle='dashdot',label='Im{C(6,7)}',color='red')
ax[2].plot(sol.times,np.imag(coherencias['6,8']),linestyle='dashdot',label='Im{C(6,8)}',color='red',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['6,9']),linestyle='dashdot',label='Im{C(6,9)}',color='cyan')
ax[2].plot(sol.times,np.imag(coherencias['6,10']),linestyle='dashdot',label='Im{C(6,10)}',color='cyan',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['6,11']),linestyle='dashdot',label='Im{C(6,11)}',color='magenta')

ax[3].plot(sol.times,sol.expect[1],label=nomb[1][0],color=colors[0])
ax[3].plot(sol.times,sol.expect[2],label=nomb[1][1],color=colors[1])
ax[3].plot(sol.times,sol.expect[3],label=nomb[1][2],color=colors[2])
ax[3].plot(sol.times,np.real(coherencias['3,4']),linestyle='dashed',label='Re{C(3,4)}',color='blue')
ax[3].plot(sol.times,np.real(coherencias['3,5']),linestyle='dashed',label='Re{C(3,5)}',color='blue',alpha=0.5)
ax[3].plot(sol.times,np.real(coherencias['3,6']),linestyle='dashed',label='Re{C(3,6)}',color='green')
ax[3].plot(sol.times,np.real(coherencias['3,7']),linestyle='dashed',label='Re{C(3,7)}',color='green',alpha=0.5)
ax[3].plot(sol.times,np.real(coherencias['3,8']),linestyle='dashed',label='Re{C(3,8)}',color='pink')
ax[3].plot(sol.times,np.real(coherencias['3,9']),linestyle='dashed',label='Re{C(3,9)}',color='pink',alpha=0.5)
ax[3].plot(sol.times,np.real(coherencias['3,10']),linestyle='dashed',label='Re{C(3,10)}',color='red')
ax[3].plot(sol.times,np.real(coherencias['3,11']),linestyle='dashed',label='Re{C(3,11)}',color='red',alpha=0.5)

ax[3].plot(sol.times,np.imag(coherencias['3,4']),linestyle='dashdot',label='Im{C(3,4)}',color='blue')
ax[3].plot(sol.times,np.imag(coherencias['3,5']),linestyle='dashdot',label='Im{C(3,5)}',color='blue',alpha=0.5)
ax[3].plot(sol.times,np.imag(coherencias['3,6']),linestyle='dashdot',label='Im{C(3,6)}',color='green')
ax[3].plot(sol.times,np.imag(coherencias['3,7']),linestyle='dashdot',label='Im{C(3,7)}',color='green',alpha=0.5)
ax[3].plot(sol.times,np.imag(coherencias['3,8']),linestyle='dashdot',label='Im{C(3,8)}',color='pink')
ax[3].plot(sol.times,np.imag(coherencias['3,9']),linestyle='dashdot',label='Im{C(3,9)}',color='pink',alpha=0.5)
ax[3].plot(sol.times,np.imag(coherencias['3,10']),linestyle='dashdot',label='Im{C(3,10)}',color='red')
ax[3].plot(sol.times,np.imag(coherencias['3,11']),linestyle='dashdot',label='Im{C(3,11)}',color='red',alpha=0.5)

ax[1].set_xlabel('t')
ax[1].set_ylabel(f'N=1 exitaciones')
ax[1].legend()

ax[2].set_xlabel('t')
ax[2].set_ylabel(f'N=1 exitaciones')
ax[2].legend()

ax[3].set_xlabel('t')
ax[3].set_ylabel(f'N=1 exitaciones')
ax[3].legend()

plt.show()

fig,ax=plt.subplots()

#COHERENCIAS N=2

ax[0].plot(sol.times,np.real(coherencias['0,4']),linestyle='dashed',label='Re{C(0,4)}',color='blue')
ax[0].plot(sol.times,np.real(coherencias['1,4']),linestyle='dashed',label='Re{C(1,4)}',color='blue',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['2,4']),linestyle='dashed',label='Re{C(2,4)}',color='green')
ax[0].plot(sol.times,np.real(coherencias['3,4']),linestyle='dashed',label='Re{C(3,4)}',color='green',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['4,5']),linestyle='dashed',label='Re{C(4,5)}',color='pink')
ax[0].plot(sol.times,np.real(coherencias['4,6']),linestyle='dashed',label='Re{C(4,6)}',color='pink',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['4,7']),linestyle='dashed',label='Re{C(4,7)}',color='red')
ax[0].plot(sol.times,np.real(coherencias['4,8']),linestyle='dashed',label='Re{C(4,8)}',color='red',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['4,9']),linestyle='dashed',label='Re{C(4,9)}',color='cyan')
ax[0].plot(sol.times,np.real(coherencias['4,10']),linestyle='dashed',label='Re{C(4,10)}',color='cyan',alpha=0.5)
ax[0].plot(sol.times,np.real(coherencias['4,11']),linestyle='dashed',label='Re{C(4,11)}',color='magenta')

ax[0].plot(sol.times,np.imag(coherencias['0,1']),linestyle='dashdot',label='Im{C(0,1)}',color='blue')
ax[0].plot(sol.times,np.imag(coherencias['0,2']),linestyle='dashdot',label='Im{C(0,2)}',color='blue',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['0,3']),linestyle='dashdot',label='Im{C(0,3)}',color='green')
ax[0].plot(sol.times,np.imag(coherencias['0,4']),linestyle='dashdot',label='Im{C(0,4)}',color='green',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['0,5']),linestyle='dashdot',label='Im{C(0,5)}',color='pink')
ax[0].plot(sol.times,np.imag(coherencias['0,6']),linestyle='dashdot',label='Im{C(0,6)}',color='pink',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['0,7']),linestyle='dashdot',label='Im{C(0,7)}',color='red')
ax[0].plot(sol.times,np.imag(coherencias['0,8']),linestyle='dashdot',label='Im{C(0,8)}',color='red',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['0,9']),linestyle='dashdot',label='Im{C(0,9)}',color='cyan')
ax[0].plot(sol.times,np.imag(coherencias['0,10']),linestyle='dashdot',label='Im{C(0,10)}',color='cyan',alpha=0.5)
ax[0].plot(sol.times,np.imag(coherencias['0,11']),linestyle='dashdot',label='Im{C(0,11)}',color='magenta')


#Coherencias N=3-

ax[1].plot(sol.times,np.real(coherencias['1,2']),linestyle='dashed',label='Re{C(1,2)}',color='blue')
ax[1].plot(sol.times,np.real(coherencias['1,3']),linestyle='dashed',label='Re{C(1,3)}',color='blue',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['1,4']),linestyle='dashed',label='Re{C(1,4)}',color='green')
ax[1].plot(sol.times,np.real(coherencias['1,5']),linestyle='dashed',label='Re{C(1,5)}',color='green',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['1,6']),linestyle='dashed',label='Re{C(1,6)}',color='pink')
ax[1].plot(sol.times,np.real(coherencias['1,7']),linestyle='dashed',label='Re{C(1,7)}',color='pink',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['1,8']),linestyle='dashed',label='Re{C(1,8)}',color='red')
ax[1].plot(sol.times,np.real(coherencias['1,9']),linestyle='dashed',label='Re{C(1,9)}',color='red',alpha=0.5)
ax[1].plot(sol.times,np.real(coherencias['1,10']),linestyle='dashed',label='Re{C(1,10)}',color='cyan')
ax[1].plot(sol.times,np.real(coherencias['1,11']),linestyle='dashed',label='Re{C(1,11)}',color='cyan',alpha=0.5)

ax[1].plot(sol.times,np.imag(coherencias['1,2']),linestyle='dashdot',label='Im{C(1,2)}',color='blue')
ax[1].plot(sol.times,np.imag(coherencias['1,3']),linestyle='dashdot',label='Im{C(1,3)}',color='blue',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['1,4']),linestyle='dashdot',label='Im{C(1,4)}',color='green')
ax[1].plot(sol.times,np.imag(coherencias['1,5']),linestyle='dashdot',label='Im{C(1,5)}',color='green',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['1,6']),linestyle='dashdot',label='Im{C(1,6)}',color='pink')
ax[1].plot(sol.times,np.imag(coherencias['1,7']),linestyle='dashdot',label='Im{C(1,7)}',color='pink',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['1,8']),linestyle='dashdot',label='Im{C(1,8)}',color='red')
ax[1].plot(sol.times,np.imag(coherencias['1,9']),linestyle='dashdot',label='Im{C(1,9)}',color='red',alpha=0.5)
ax[1].plot(sol.times,np.imag(coherencias['1,10']),linestyle='dashdot',label='Im{C(1,10)}',color='cyan')
ax[1].plot(sol.times,np.imag(coherencias['1,11']),linestyle='dashdot',label='Im{C(1,11)}',color='cyan',alpha=0.5)


#---Coherencias N=4---

ax[2].plot(sol.times,np.real(coherencias['2,3']),linestyle='dashed',label='Re{C(2,3)}',color='blue')
ax[2].plot(sol.times,np.real(coherencias['2,4']),linestyle='dashed',label='Re{C(2,4)}',color='blue',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['2,5']),linestyle='dashed',label='Re{C(2,5)}',color='green')
ax[2].plot(sol.times,np.real(coherencias['2,6']),linestyle='dashed',label='Re{C(2,6)}',color='green',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['2,7']),linestyle='dashed',label='Re{C(2,7)}',color='pink')
ax[2].plot(sol.times,np.real(coherencias['2,8']),linestyle='dashed',label='Re{C(2,8)}',color='pink',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['2,9']),linestyle='dashed',label='Re{C(2,9)}',color='red')
ax[2].plot(sol.times,np.real(coherencias['2,10']),linestyle='dashed',label='Re{C(2,10)}',color='red',alpha=0.5)
ax[2].plot(sol.times,np.real(coherencias['2,11']),linestyle='dashed',label='Re{C(2,11)}',color='cyan')

ax[2].plot(sol.times,np.imag(coherencias['2,3']),linestyle='dashdot',label='Im{C(2,3)}',color='blue')
ax[2].plot(sol.times,np.imag(coherencias['2,4']),linestyle='dashdot',label='Im{C(2,4)}',color='blue',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['2,5']),linestyle='dashdot',label='Im{C(2,5)}',color='green')
ax[2].plot(sol.times,np.imag(coherencias['2,6']),linestyle='dashdot',label='Im{C(2,6)}',color='green',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['2,7']),linestyle='dashdot',label='Im{C(2,7)}',color='pink')
ax[2].plot(sol.times,np.imag(coherencias['2,8']),linestyle='dashdot',label='Im{C(2,8)}',color='pink',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['2,9']),linestyle='dashdot',label='Im{C(2,9)}',color='red')
ax[2].plot(sol.times,np.imag(coherencias['2,10']),linestyle='dashdot',label='Im{C(2,10)}',color='red',alpha=0.5)
ax[2].plot(sol.times,np.imag(coherencias['2,11']),linestyle='dashdot',label='Im{C(2,11)}',color='cyan')

# colors = plt.cm.jet(np.linspace(0,1,len(nomb[2])))
# ax[2].plot(sol.times,sol.expect[4],label=nomb[2][0],color=colors[0])
# ax[2].plot(sol.times,sol.expect[5],label=nomb[2][1],color=colors[1])
# ax[2].plot(sol.times,sol.expect[6],label=nomb[2][2],color=colors[2])
# ax[2].plot(sol.times,sol.expect[7],label=nomb[2][3],color=colors[3])
# ax[2].set_xlabel('t')
# ax[2].set_ylabel(f'N=2 exitaciones')
# ax[2].legend()

# ax[-1].plot(sol.times,sol.expect[-5],color='black',label='<0.5(sz1+sz2)>')
# ax[-1].plot(sol.times,sol.expect[-4],color='red',label='<sz1>')
# ax[-1].plot(sol.times,sol.expect[-3],color='orange',label='<sz2>')
# ax[-1].plot(sol.times,sol.expect[-2],color='blue',label='<sx1>')
# ax[-1].plot(sol.times,sol.expect[-1],color='cyan',label='<sx2>')
# ax[-1].set_xlabel('t')
# ax[-1].set_ylabel('U.A.')
# ax[-1].legend()

plt.show()'''