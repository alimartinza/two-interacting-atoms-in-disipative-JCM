import numpy as np
delta=1
k=1
n=1
x=1
J=1
g=1
a=  {-1j*g*np.sqrt(2*n)*(-2*np.real(rho[1])),
     1j*(g*np.sqrt(2*n)*rho[0] + (-2*J + 2*k + delta + (1 - 2*n)*x)*rho[1] + g*(np.sqrt(2*n-2)*rho[2] - np.sqrt(2*n)*rho[4])), 
  1j*(g*np.sqrt(2*n-2)*rho[1] + 2*(delta + 2*x - 2*n*x)*rho[2] - g*np.sqrt(2*n)*rho[5]),
    -1j*((2*J - 2*k - delta - x + 2*n*x)*rho[3] + g*np.sqrt(2*n)*rho[6]),
 -1j*((-2*J + 2*k + delta + x - 2*n*x)*np.conj(rho[1]) + g*(np.sqrt(2*n-2)*np.conj(rho[2]) + np.sqrt(2*n)*(rho[0] - rho[4]))), 
  1j*g*(np.sqrt(2*n)*np.conj(rho[1]) - np.sqrt(2*n-2)*np.conj(rho[5]) - np.sqrt(2*n)*rho[1] + np.sqrt(2*n-2)*rho[5]),
    -1j*(np.sqrt(2)*g*np.sqrt(n)*rho[2] - np.sqrt(2)*g*np.sqrt(n-1)*rho[4] +(-2*J + 2*k - delta - 3*x + 2*n*x)*rho[5] + np.sqrt(2)*g*np.sqrt(n-1)*rho[7]), 
-1j*np.sqrt(2)*g*(np.sqrt(n)*rho[3] + np.sqrt(n-1)*rho[8]),
-1j*(np.sqrt(2)*g*np.sqrt(n-1)*np.conj(rho[1]) + 2*(delta + 2*x - 2*n*x)*np.conj(rho[2]) - np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[5])), 
  1j*(np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[2]) - (2*J - 2*k + delta + 3*x - 2*n*x)*np.conj(rho[5] + np.sqrt(2)*g*np.sqrt(n-1)*(-rho[4] + rho[7]))), 
  1j*np.sqrt(2)*g*np.sqrt(-1 + n)*(np.conj(rho[5]) - rho[5]), -1j*(np.sqrt(2)*g*np.sqrt(n-1)*rho[6] + (2*J - 2*k + delta + 3*x - 2*n*x)*rho[8]),
 1j*((2*J - 2*k - delta - x + 2*n*x)*np.conj(rho[3]) + np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[6])), 
  1j*np.sqrt(2)*g*(np.sqrt(n)*np.conj(rho[3]) + np.sqrt(n-1)*np.conj(rho[8])), 
  1j*(np.sqrt(2)*g*np.sqrt(n-1)*np.conj(rho[6]) + (2*J - 2*k + delta + 3*x - 2*n*x)*np.conj(rho[8])),
                                          0
}

def rho_punto(rho):
    return np.array([-1j*g*np.sqrt(2*n)*(-2*np.real(rho[1])),
     1j*(g*np.sqrt(2*n)*rho[0] + (-2*J + 2*k + delta + (1 - 2*n)*x)*rho[1] + g*(np.sqrt(2*n-2)*rho[2] - np.sqrt(2*n)*rho[4])), 
  1j*(g*np.sqrt(2*n-2)*rho[1] + 2*(delta + 2*x - 2*n*x)*rho[2] - g*np.sqrt(2*n)*rho[5]),
    -1j*((2*J - 2*k - delta - x + 2*n*x)*rho[3] + g*np.sqrt(2*n)*rho[6]),
 -1j*((-2*J + 2*k + delta + x - 2*n*x)*np.conj(rho[1]) + g*(np.sqrt(2*n-2)*np.conj(rho[2]) + np.sqrt(2*n)*(rho[0] - rho[4]))), 
  1j*g*(np.sqrt(2*n)*np.conj(rho[1]) - np.sqrt(2*n-2)*np.conj(rho[5]) - np.sqrt(2*n)*rho[1] + np.sqrt(2*n-2)*rho[5]),
    -1j*(np.sqrt(2)*g*np.sqrt(n)*rho[2] - np.sqrt(2)*g*np.sqrt(n-1)*rho[4] +(-2*J + 2*k - delta - 3*x + 2*n*x)*rho[5] + np.sqrt(2)*g*np.sqrt(n-1)*rho[7]), 
-1j*np.sqrt(2)*g*(np.sqrt(n)*rho[3] + np.sqrt(n-1)*rho[8]),
-1j*(np.sqrt(2)*g*np.sqrt(n-1)*np.conj(rho[1]) + 2*(delta + 2*x - 2*n*x)*np.conj(rho[2]) - np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[5])), 
  1j*(np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[2]) - (2*J - 2*k + delta + 3*x - 2*n*x)*np.conj(rho[5] + np.sqrt(2)*g*np.sqrt(n-1)*(-rho[4] + rho[7]))), 
  1j*np.sqrt(2)*g*np.sqrt(-1 + n)*(np.conj(rho[5]) - rho[5]), -1j*(np.sqrt(2)*g*np.sqrt(n-1)*rho[6] + (2*J - 2*k + delta + 3*x - 2*n*x)*rho[8]),
 1j*((2*J - 2*k - delta - x + 2*n*x)*np.conj(rho[3]) + np.sqrt(2)*g*np.sqrt(n)*np.conj(rho[6])), 
  1j*np.sqrt(2)*g*(np.sqrt(n)*np.conj(rho[3]) + np.sqrt(n-1)*np.conj(rho[8])), 
  1j*(np.sqrt(2)*g*np.sqrt(n-1)*np.conj(rho[6]) + (2*J - 2*k + delta + 3*x - 2*n*x)*np.conj(rho[8])),
  0])

r1[0]== r2[0]== r3[0]== r4[0]== r5[0]== r6[0]== r7[0]== r8[0]== r9[0]== r10[0]== r11[0]== r12[0]==
 r13[0]== r14[0]== r15[0]== r16[0]== r17[0]== r18[0]== r19[0]== r20[0]== r21[0]== r22[0]== r23[0]== r24[0]==
 r25[0]== r26[0]== r27[0]== r28[0]== r29[0]== r30[0]== r31[0]== r32[0]== r33[0]== r34[0]== r35[0]== r36[0]==
 r37[0]== r38[0]== r39[0]== r40[0]== r41[0]== r42[0]== r43[0]== r44[0]== r45[0]== r46[0]== r47[0]== r48[0]==
 r49[0]== r50[0]== r51[0]== r52[0]== r53[0]== r54[0]== r55[0]== r56[0]== r57[0]== r58[0]== r59[0]== r60[0]==
 r61[0]== r62[0]== r63[0]== r64[0]== r65[0]== r66[0]== r67[0]== r68[0]== r69[0]== r70[0]== r71[0]== r72[0]==
 r73[0]== r74[0]== r75[0]== r76[0]== r77[0]== r78[0]== r79[0]== r80[0]== r81[0]== r82[0]== r83[0]== r84[0]==
 r85[0]== r86[0]== r87[0]== r88[0]== r89[0]== r90[0]== r91[0]== r92[0]== r93[0]== r94[0]== r95[0]== r96[0]==
 r97[0]== r98[0]== r99[0]== r100[0]== r101[0]== r102[0]== r103[0]== r104[0]== r105[0]== r106[0]== r107[0]== r108[0]==
 r109[0]== r110[0]== r111[0]== r112[0]== r113[0]== r114[0]== r115[0]== r116[0]== r117[0]== r118[0]== r119[0]== 
  r120[0]==
 r121== r122[0]== r123[0]== r124[0]== r125[0]== r126[0]== r127[0]== r128[0]== r129[0]== r130[0]== r131[0]== 
  r132[0]==
 r133[0]== r134[0]== r135[0]== r136[0]== r137[0]== r138[0]== r139[0]== r140[0]== r141[0]== r142[0]== r143[0]== 
  r144[0]

{{r1[t], r2[t], r3[t], r4[t], r5[t], r6[t], r7[t], r8[t], r9[t], r10[t], r11[t], r12[t]},
 {r13[t], r14[t], r15[t], r16[t], r17[t], r18[t], r19[t], r20[t], r21[t], r22[t], r23[t], r24[t]},
 {r25[t], r26[t], r27[t], r28[t], r29[t], r30[t], r31[t], r32[t], r33[t], r34[t], r35[t], r36[t]},
 {r37[t], r38[t], r39[t], r40[t], r41[t], r42[t], r43[t], r44[t], r45[t], r46[t], r47[t], r48[t]},
 {r49[t], r50[t], r51[t], r52[t], r53[t], r54[t], r55[t], r56[t], r57[t], r58[t], r59[t], r60[t]},
 {r61[t], r62[t], r63[t], r64[t], r65[t], r66[t], r67[t], r68[t], r69[t], r70[t], r71[t], r72[t]},
 {r73[t], r74[t], r75[t], r76[t], r77[t], r78[t], r79[t], r80[t], r81[t], r82[t], r83[t], r84[t]},
 {r85[t], r86[t], r87[t], r88[t], r89[t], r90[t], r91[t], r92[t], r93[t], r94[t], r95[t], r96[t]},
 {r97[t], r98[t], r99[t], r100[t], r101[t], r102[t], r103[t], r104[t], r105[t], r106[t], r107[t], r108[t]},
 {r109[t], r110[t], r111[t], r112[t], r113[t], r114[t], r115[t], r116[t], r117[t], r118[t], r119[t], 
  r120[t]},
 {r121, r122[t], r123[t], r124[t], r125[t], r126[t], r127[t], r128[t], r129[t], r130[t], r131[t], 
  r132[t]},
 {r133[t], r134[t], r135[t], r136[t], r137[t], r138[t], r139[t], r140[t], r141[t], r142[t], r143[t], 
  r144[t]}}