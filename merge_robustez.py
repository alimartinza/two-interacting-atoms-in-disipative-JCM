import numpy as np

start1=0
fin1=1
steps1=406

start2=0
fin2=1
steps2=50

x1=np.linspace(start1,fin1,steps1)
x2=np.linspace(start2,fin2,steps2)
xnew=np.zeros(len(x1)+len(x2))
j=0
for i in range(len(xnew)):
    if x1[i]>x2[j]:
        np.insert(x1,i,x2[j])
        j+=1
    else:
        None
