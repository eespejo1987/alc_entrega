"""
alc.py (SOLO lo necesario para el Ejercicio 4 - homografía en la cancha).
Solo usa de numpy: np.array, np.zeros, np.ones, np.eye, np.sqrt (lista permitida del enunciado).

"""
import numpy as np

def norma(x, p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    x = np.array(x)
    # 1. Norma 1
    if p == 1:
        total = 0
        for i in range(len(x)):
            total += abs(x[i])
        return total

    # 2. Norma infinito
    if p == "inf":
        max_val = 0
        for i in range(len(x)):
            if abs(x[i]) > max_val:
                max_val = abs(x[i])
        return max_val

    # 3. Caso lista o tupla
    if p == [1, "inf"] or p == (1, "inf"):
        return [norma(x, 1), norma(x, "inf")]

    # 4. Cualquier otra norma p (p=2, p=3, etc.)
    total = 0
    for i in range(len(x)):
        total += abs(x[i]) ** p
    return total ** (1 / p)

def fA(k,A,v):
    w_0 = matriz_x_matriz(A,v)

    w_1 = 0

    if norma(w_0,2) > 0:
        #np.linalg.norm(w_0, ord=2)
        #w_1 = w_0/np.linalg.norm(w_0,ord=2)
        w_normalizado= norma(w_0,2)
        w_1 = w_0/w_normalizado
        for i in range(k-1):
            w_0 = matriz_x_matriz(A,w_1)
            w_normalizado= norma(w_0,2)
            w_1 = w_0/w_normalizado
    return w_1


def metpot2k(A, tol=1e-8,K=1000):
    n = A.shape[0]
    v = np.random.random(n)
    w = fA(2,A,v)
    e = matriz_x_matriz(w.T,v)
    k = 0
    while np.abs(e-1) > tol and k < K:
        v = w
        w = fA(2,A,v)
        e = matriz_x_matriz(w.T,v)
        k += 1
    l = matriz_x_matriz(w.T,matriz_x_matriz(A,w))
    return v, l, k
# ------------------------------------------------------------------
# PARTE 1: cuentas básicas hechas a mano
# ------------------------------------------------------------------
def multiplicar_matrices(A,B):
    """
    Dada una matriz A y B las multiplica
    """
    n, m = A.shape
    m2, p = B.shape
    if m!=m2:
        return ValueError("las dimensiones de las matrices no coinciden")
    
    AB = np.zeros((n,p))
    for i in range(n):
        for j in range(p): # uso np.sum para optimizar
            AB[i,j] = np.sum(A[i, :]* B[:,j])
    
    return AB

def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res


def _aplanar(x):
    """Pasa un vector (n,), (n,1) o (1,n) -- o un escalar -- a un vector 1D, con bucles."""
    x = np.array(x, dtype=float)
    f = len(x.shape)
    if f == 0:
        R = np.zeros(1)
        R[0] = float(x)
        return R
    cant = 1
    for d in x.shape:
        cant = cant * d
    R = np.zeros(cant)
    if f == 1:
        for i in range(cant):
            R[i] = x[i]
    else:
        k = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                R[k] = x[i, j]
                k = k + 1
    return R
#version APTA para tp
def diagRH(A, tol=1e-15, K=1000):
    """Diagonaliza una matriz simétrica A = S D S^t con reflexiones de Householder:
    método de la potencia para el autovector principal, una reflexión que lo lleva a e1,
    y se repite sobre el bloque que queda (recursión). Devuelve (S, D) o None si A no es simétrica."""
    if not(esSimetrica(A)):
        return None
    n = A.shape[0]
    v1, l1, _ = metpot2k(A, tol, K)
    v1 = _aplanar(v1)
    lam = _aplanar(l1)[0]
    e1 = vector_canonico(n, 0)
    u = np.zeros(n)
    for i in range(n):
        u[i] = e1[i] - v1[i]
    u_col = np.zeros((n, 1))
    u_fila = np.zeros((1, n))
    for i in range(n):
        u_col[i, 0] = u[i]
        u_fila[0, i] = u[i]
    u_normalizado = norma(u, 2)
    if u_normalizado < 1e-14:            # v1 ya es e1: no hace falta reflejar
        H_v1 = np.eye(n)
    else:
        H_v1 = np.eye(n) - 2 * (matriz_x_matriz(u_col, u_fila) / (u_normalizado ** 2))
    if n == 2:
        S = H_v1
        D = matriz_x_matriz(H_v1, matriz_x_matriz(A, transpuesta(H_v1)))
    else:
        B = matriz_x_matriz(H_v1, matriz_x_matriz(A, transpuesta(H_v1)))
        AA = B[1:n, 1:n]
        m = n - 1
        for i in range(m):                # saca la asimetría de redondeo
            for j in range(i + 1, m):
                prom = (AA[i, j] + AA[j, i]) / 2 #esto habria q aclarar q onda la version anterior no la tenia. 
                AA[i, j] = prom
                AA[j, i] = prom
        SS, DD = diagRH(AA, tol, K)
        D = np.zeros((n, n))
        D[0][0] = lam
        D[1:n, 1:n] = DD
        SSS = np.zeros((n, n))
        SSS[0][0] = 1
        SSS[1:n, 1:n] = SS
        S = matriz_x_matriz(H_v1, SSS)
    # Ajusto los elementos que no son diagonales
    for x in range(0, len(D), 1):
        for y in range(0, len(D), 1):
            if y != x:
                D[x][y] = 0.0
    return S, D

"""Auxiliares de estos ejercicios"""


def vector_canonico(n,i):
    x = [0.0 for _ in range(n)]
    x[i]=1.0
    x=np.array(x)
    return x

import numpy as np

def esSimetrica(A, tol=1e-10):
    n, m = A.shape
    if n != m:
        return False
    
    for i in range(n):
        for j in range(m):
            if abs(A[i,j] - A[j,i]) > tol:
                return False
                
    return True
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# PARTE 2: SVD reducida
# ------------------------------------------------------------------
# svd_reducida usa estos nombres (son los del labo). Se apoyan en las funciones de la PARTE 1:
 

def matriz_por_vector(A, v):
    """Recibe una matriz A (n x m) y un vector v (m).
    Retorna un vector w (n) resultado de multiplicar A con v (w = A * v).
    """
    n, m = A.shape
    w = np.zeros(n) 
    
    for fila in range(n): 
        suma = 0 
        for columna in range(m):
            suma = suma + A[fila, columna] * v[columna]
        w[fila] = suma
        
    return w

def matriz_x_matriz(A,B):
    # Si Dim(A)=1 o Dim(B)=1, lo convierte a vector columna.
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    m, n = A.shape
    i, j = B.shape

    if i!=n:
        return None
    res=np.array(np.zeros((m,j)))

    for x in range(0,j,1):
        elem=matriz_por_vector(A,B[:,x])
        res[:,x]=elem
    return res


def svd_reducida(A,k="max",tol=1e-15):
    m, n = A.shape
    k_valoresSingulares=[]
    if m>=n:
        C = matriz_x_matriz(transpuesta(A),A)
        V, D = diagRH(C)
        f,c =D.shape
        for j in range(f):
            autovalor= D[j,j]
            sigma = abs(autovalor)
            if (sigma>tol):
                k_valoresSingulares.append(np.sqrt(sigma))
            else:
                break
        r= len(k_valoresSingulares)
        if k=="max" :
            c = r
        else:
            c = min(k,r)
        k_valoresSingulares=k_valoresSingulares[:c]
        hatSig = np.array(k_valoresSingulares)
        hatV = V[:, :c]
        hatU= np.zeros((m,c))
        B = multiplicar_matrices(A, hatV)
        for i in range(c):
            hatU[:,i] = B[:,i]/hatSig[i]
            hatU[0:m,i] = hatU[0:m,i]/norma((hatU[0:m,i]),2)
    else:
        C = matriz_x_matriz(A,transpuesta(A))
        U, D = diagRH(C)
        f,c =D.shape
        for j in range(f):
            autovalor= D[j,j]
            sigma = np.sqrt(abs(autovalor))
            if (sigma>tol):
                k_valoresSingulares.append(sigma)
            else:
                break
        r= len(k_valoresSingulares)
        if k=="max" :
            c = r
        else:
            c = min(k,r)
        k_valoresSingulares=k_valoresSingulares[:c]
        hatSig = np.array(k_valoresSingulares)
        hatU = U[:, :c]
        hatV= np.zeros((n,c))
        B = matriz_x_matriz(transpuesta(A), hatU)
        for i in range(c):
            hatV[:,i] = B[:,i]/hatSig[i]
            hatV[0:n,i] = hatV[0:n,i]/norma((hatV[0:n,i]),2)
    return hatU, hatSig, hatV


# ------------------------------------------------------------------
# PARTE 3: inversa de una matriz (para ir de la foto a la cancha)
# ------------------------------------------------------------------
def detMatrizTriangular(U, tol=1e-10):  #funcion auxiliar para ver si U es sing
    determinante = 1 
    for i in range (U.shape[0]):
        determinante = determinante*U[i,i]
    if abs(determinante) < tol : 
        return 0 
    return determinante

def esCuadrada(A)->bool:
    col=len(A[0])

    #verifico la cantidad de columnas por fila
    for x in range(1,len(A),1):
        if len(A[x])!=col:
            return False
    return len(A)==col


def calculaLU(A): # pasa los test locales
    if A is None:
        return None, None, 0
        
    cant_op = 0
    n = A.shape[0]  # filas
    m = A.shape[1]  # columnas
    
    if m != n:
        return None, None, 0
        
    U = A.copy()
    L = np.eye(n, n)
    
    for i in range(n - 1):
        if U[i, i] == 0:
            return None, None, 0
            
        pivote = U[i, i]
        for k in range(i + 1, n):  # filas por debajo del pivote
            factor = U[k, i] / pivote
            cant_op += 1  # 1 división
            L[k, i] = factor  # guardo el factor en L
            
            U[k, i] = 0  # anulamos explícitamente sin gastar operaciones aritméticas
            
            for j in range(i + 1, n):  # <--- Empieza en i + 1
                U[k, j] = U[k, j] - factor * U[i, j]
                cant_op += 2  # 1 multiplicación y 1 resta
                
    return L, U, cant_op

def res_tri(A, b, inferior=True):
    """
    Resuelve Ax = b cuando A es triangular (inferior si inferior=True,
    superior si inferior=False), por sustitución hacia adelante/atrás.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
 
    if inferior:
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
    else:
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
 
    return x

def inversa(A):    
    """
    Para cada elemento de la Identidad resuelve el sistema Ax = Ei. Con Ei vector canónico correspondiente a la posición i. 
     
    """
    #Obtengo dimensión de la matriz original
    n=A.shape[0]
    #creo matriz de 0
    res=np.zeros((n,n))
    I=np.eye((n))
    L, U, co= calculaLU(A) #co =cantidad de operaciones
    #Det A = Det L * Det U
    det_L=detMatrizTriangular(L)
    det_U=detMatrizTriangular(U)

    if (det_L*det_U)==0:
        return None
    for x in range(0,len(I),1):
        # LUx=b
        Up = res_tri(L,I[:,x],True) # Ly=b
        v = res_tri(U,Up,False)     # Ux=y   
        res[:,x]=v
    return res

def normalizar_puntos(pts):
    """Hartley (sirve para puntos 2D o 3D): traslada el centro de los puntos al origen y los
    escala para que su distancia promedio al origen sea sqrt(d) (d = 2 o 3).
    Devuelve (pts_nuevos, T) con T la matriz (d+1)x(d+1) que hace ese cambio."""
    n = pts.shape[0]
    d = pts.shape[1]
    centro = np.zeros(d)
    for i in range(n):
        for j in range(d):
            centro[j] = centro[j] + pts[i, j] / n
    dist_total = 0.0
    for i in range(n):
        s2 = 0.0
        for j in range(d):
            s2 = s2 + (pts[i, j] - centro[j]) ** 2
        dist_total = dist_total + np.sqrt(s2)
    s = np.sqrt(d) / (dist_total / n)
    T = np.eye(d + 1)
    for j in range(d):
        T[j, j] = s
        T[j, d] = -s * centro[j]
    pts_nuevos = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            pts_nuevos[i, j] = s * (pts[i, j] - centro[j])
    return pts_nuevos, T

def armar_Q(img_pts, obj_pts):
    """Tabla Q del DLT. obj_pts puede ser (n x 3) -> Q de 2n x 12 (ejercicio 2),
    o (n x 2) -> Q de 2n x 9 (ejercicios 3 y 4). Cada punto P = (coordenadas, 1) aporta 2 filas:
        [ P^t   0    -u P^t ]
        [  0   P^t   -v P^t ]"""
    n = obj_pts.shape[0]
    d = obj_pts.shape[1]
    k = d + 1
    Q = np.zeros((2 * n, 3 * k))
    for i in range(n):
        P = np.ones(k)
        for j in range(d):
            P[j] = obj_pts[i, j]
        u = img_pts[i, 0]; v = img_pts[i, 1]
        for j in range(k):
            Q[2 * i, j] = P[j]
            Q[2 * i, 2 * k + j] = -u * P[j]
            Q[2 * i + 1, k + j] = P[j]
            Q[2 * i + 1, 2 * k + j] = -v * P[j]
    return Q


def obtenerQ2d(imgPts, objPts):
    """Ecuación 11 (puntos coplanares): Q de 2n x 9."""
    return armar_Q(imgPts, objPts)

def autovector_menor(Q, filas, cols):
    """Resuelve min ||Q h||^2 con ||h|| = 1 usando la SVD reducida de Q:
    h = columna de V asociada al MENOR valor singular, acomodada como matriz filas x cols (por filas)."""
    m = Q.shape[0]; p = Q.shape[1]
    # Con menos filas que columnas (ej. 4 puntos: 8 x 9) la SVD reducida perdería el vector
    # del núcleo. Se completa con filas de ceros (no cambia Q^t Q, ni la solución).
    if m < p:
        Qc = np.zeros((p, p))
        for i in range(m):
            for j in range(p):
                Qc[i, j] = Q[i, j]
        Q = Qc
    # tol = -1: NO se descarta ningun valor singular. Con la tolerancia por defecto (1e-15), si Q tiene
    # un valor singular casi nulo (datos sin ruido, o solo 4 puntos) svd_reducida lo tira como si fuera
    # cero y se perderia justamente el vector del nucleo, que es la solucion que buscamos.
    U, S, V = svd_reducida(Q, "max", -1.0)
    k = S.shape[0] - 1                      # svd_reducida ordena de mayor a menor: el menor es el último
    R = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            R[i, j] = V[i * cols + j, k]
    return R

def resolver_H(Q):
    """Ejercicio 3/4: devuelve H (3x3) = vector singular de Q asociado al menor valor singular."""
    return autovector_menor(Q, 3, 3)

def estimar_homografia(img_pts, obj_pts):
    """Receta completa: normalizar, resolver, deshacer la normalización."""
    img_n, T_img = normalizar_puntos(img_pts)
    obj_n, T_obj = normalizar_puntos(obj_pts)
    H_n = resolver_H(obtenerQ2d(img_n, obj_n))
    H = multiplicar_matrices(inversa(T_img), multiplicar_matrices(H_n, T_obj))
    for i in range(3):                        # dejar H[2,2] = 1 (solo prolijidad)
        for j in range(3):
            if (i, j) != (2, 2):
                H[i, j] = H[i, j] / H[2, 2]
    H[2, 2] = 1.0
    return H

def aplicar_homografia(H, pts):
    """Lleva cada punto (a,b) a otro lugar: q = H*(a,b,1) y se DIVIDE por la 3ra coordenada.
    Con H va cancha->foto; con inversa(H) va foto->cancha."""
    n = pts.shape[0]
    R = np.zeros((n, 2))
    for i in range(n):
        x = H[0, 0] * pts[i, 0] + H[0, 1] * pts[i, 1] + H[0, 2]
        y = H[1, 0] * pts[i, 0] + H[1, 1] * pts[i, 1] + H[1, 2]
        z = H[2, 0] * pts[i, 0] + H[2, 1] * pts[i, 1] + H[2, 2]
        R[i, 0] = x / z
        R[i, 1] = y / z
    return R

def error_reproyeccion(H, img_pts, obj_pts):
    """Qué tan lejos (en píxeles, en promedio cuadrático) caen los puntos proyectados con H
    respecto de donde vos los marcaste. Si da pocos píxeles, H está bien."""
    pred = aplicar_homografia(H, obj_pts)
    n = obj_pts.shape[0]
    suma = 0.0
    for i in range(n):
        dx = pred[i, 0] - img_pts[i, 0]; dy = pred[i, 1] - img_pts[i, 1]
        suma = suma + dx * dx + dy * dy
    return np.sqrt(suma / n)

def circulo(centro, radio, N=50):
    """Puntos de un círculo en METROS, sin usar cos/sin (no están permitidas).
    Usa que con t entre -1 y 1, el punto ((1-t^2)/(1+t^2), 2t/(1+t^2)) recorre la mitad
    derecha de un círculo de radio 1; la mitad izquierda es la derecha espejada."""
    cant = 2 * N + 1
    R = np.zeros((2 * cant, 2))
    for i in range(cant):
        t = (i - N) / N
        x = (1 - t * t) / (1 + t * t)
        y = 2 * t / (1 + t * t)
        R[i, 0] = centro[0] + radio * x
        R[i, 1] = centro[1] + radio * y
        R[cant + i, 0] = centro[0] - radio * x
        R[cant + i, 1] = centro[1] + radio * y
    return R

def recta_vertical(c, ancho, N=2):
    """Puntos de la recta 'U = c' para V entre 0 y ancho (paralela a la línea de fondo)."""
    R = np.zeros((N + 1, 2))
    for i in range(N + 1):
        R[i, 0] = c
        R[i, 1] = ancho * i / N
    return R
