from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm



# def information():

# print(tensor(sigmaz(),sigmax()))

a=np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=float).T
np.savetxt('a.txt',a,'%.3f',delimiter=' ',header='t fg_tot fg_ab ')