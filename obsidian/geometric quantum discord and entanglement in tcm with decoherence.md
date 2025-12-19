Hacen el mismo modelo que yo creo, pero en la base (no simetrizada) encuentran una solucion unitaria mas facil... no se si es tema de no simetrizar los estados pero bueno. Despues usan un metodo medio falopa para poner la decoherencia, que llaman "decoherencia intrinseca", donde citan un paper [Milburn]() y dicen que la ecuacion maestra suponendo que el sistema evoluciona continuamente con una secuencia estocastica de matrices unitarias identicas. Esta ecuacion maestra es
$\frac{d\rho(t)}{dt}=-i[H,\rho(t)]-\frac{1}{2\gamma}[H,[H,\rho(t)]]$
Que tiene una solucion exacta
$\rho(t)=\sum_{k=0}^\infty \frac{t^k}{\gamma k!}M^k(t)\rho(0)M^{\dagger k}(t)$
con $M^k(t)=H^ke^{iHt}e^{-\frac{t}{2\gamma}H^2}$
que se puede escribir tambien en la base de autoestados de energia como
$\rho(t)=\sum_{mn}exp[-\frac{t}{2\gamma}(E_m-E_n)^2-i(E_m-E_n)t]\bra{\psi_m}\rho(0)\ket{\psi_m}\bra{\psi_n}$
No entiendo como esto es una matriz pero bueno, le creemos antes de hacer las cuentas.

Hasta aca no hicieron mucho mas que encontrar la solucion unitaria del subespacio 4x4. Despues usan esta falopeada de ecuacion maestra, y trazan sobre el estado de Fock $\ket{n}$ y utilizan una condicion inicial entrelazada (el estado simetrico) y obtienen que la matriz densidad reducida es bastante sencilla
$\rho(t)=\begin{pmatrix} \rho_{11} & 0 & 0 & 0 \\ 0 & \rho_{22} & \rho_{23} & 0 \\ 0 & \rho_{32} & \rho_{33} & 0 \\ 0 & 0 & 0 & \rho_{44} \end{pmatrix}$
y los coeficientes son bastante sencillos dentro de todo.
De aca encuentran expresiones para el geometric quantum discord, y para la concurrence. 
Hacen un pequeño analisis numerico, pero que no pasa de 1 parrafo, y son varias imagenes. Solo dicen que oscila el GQD, y que las curvas de concurrence son similares, pero que presentan SDE. No dicen porque, ni cuando ni nada. Solo que para paramertos mas altos de interaccion dipolar-dipolar, la concurrence llega a estados asintoticos con mayor concurrence.