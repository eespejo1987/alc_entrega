import numpy as np
import matplotlib as plt
#from todos_modulos import *

def agregar_elemento(v):
    """Incrementa en 1 la dimensión del vector, con 1 en la última posición"""
    w = np.ones(len(v) + 1)
    w[:-1] = v
    return w

def ordenar_factores(M):
    """Multiplica M por una matriz que permuta y cambia de signo"""
    p=np.array([[0.,0.,1],[-1,0.,0.],[0.,-1,0.]])
    M=matriz_x_matriz(p,M)
    return M

def construye_T(dimension):
    "Construye la matriz T para ordenar las ecuaciones de u y v"
    I = np.eye(dimension)
    O = np.zeros((dimension, dimension))
    T = np.vstack([np.hstack([I, O, O]),
        np.hstack([O, O, O]),
        np.hstack([O, I, O]),
        np.hstack([O, O, O]),
        np.hstack([I, O, O]),
        np.hstack([O, O, I])])
    return T


def construye_Ai(M,T):
    """construye las matrices de la forma [[U V W 1 0 0 0 0 -Uu -Vu -Wu -u][0 0 0 0 U V W 1 -Uv -Vv -Wv -v]]"""
    dim = M.shape[1]
    z = M.reshape(3*dim, 1)
    A_i = matriz_x_matriz(T,z).reshape(2, 3*dim)
    return A_i

## EJERCICIO 2 A ##
def obtenerQ3d(datos_2d_imagen, datos_3d):
    """"Recibe un elemento 2d y todo el dataset 3D"""
    T=construye_T(4)
    for j in range(0,len(datos_3d),1):
        coordenadas_3d=datos_3d[j]
        coordenadas_3d=agregar_elemento(coordenadas_3d)
        coordenadas_3d=coordenadas_3d.reshape(1, -1)
        par_uv=datos_2d_imagen[j*2:j*2+2] #elijo el par
        par_uv=agregar_elemento(par_uv)
        par_uv=par_uv.reshape(-1,1)
        base_Aj=matriz_x_matriz(par_uv,coordenadas_3d)
        base_Aj=ordenar_factores(base_Aj)
        Aj=construye_Ai(base_Aj,T)
        if j==0:
            A=Aj
        else:
            A=np.vstack((A, Aj)) 
    return A
# EJERCICIO 2B
def resolverDLT3D(Q):
    #construimos A=QtQ
    Qt= transpuesta(Q)
    A = matriz_x_matriz(Qt,Q)
    # diagonalizamos A
    S,D= diagRH(A)
    # busco el menor autovalor
    indice = 0
    for i in range(1,D.shape[0]):
        if D[i,i] < D[indice,indice]:
            indice = i
    # autovector correspondiente
    m = S[:,indice]
    # normalizamos el autovector
    m = m/norma(m,2)
    # construimos la matriz de 3x4
    M = m.reshape(3,4)
    return M

# EJERCICIO 2c
def devolverMatricesCalibracion3d(M):
    # Divido M (3x4) en A (3x3) y b (3,)
    A = M[:, :3]
    b = M[:, 3]

    # Factorización QR de A inversa
    A_inv = inversa(A)
    Q1, R1 = calculaQR(A_inv)

    # Recuperamos K y R provisionales
    K0 = inversa(R1)
    R0 = transpuesta(Q1)

    # Normalizamos K para que K[2,2] = 1
    c = K0[2, 2]
    K = K0 / c

    # Determinamos el signo del determinante de R0
    det = np.linalg.det(R0)

    if det > 0:
        s = 1
    else:
        s = -1

    # Matriz de rotación con determinante positivo
    R = s * R0

    # Factor de escala de la proyección
    escala = c * s

    # Calculamos la traslación
    t = s * matriz_por_vector(R1, b)

    return R, t, K
## EJERCICIO 3  ##
def obtenerQ2d(datos_2d_imagen, datos_2d_objeto):
    """Construye Q para las correspondencias entre puntos 2D."""
    T = construye_T(3)
    for j in range(0, len(datos_2d_objeto), 1):
        coordenadas_2d = datos_2d_objeto[j]
        coordenadas_2d = agregar_elemento(coordenadas_2d)
        coordenadas_2d = coordenadas_2d.reshape(1, -1)
        par_uv = datos_2d_imagen[j*2:j*2+2]
        par_uv = agregar_elemento(par_uv)
        par_uv = par_uv.reshape(-1, 1)
        base_Aj = matriz_x_matriz(par_uv,coordenadas_2d)
        base_Aj = ordenar_factores(base_Aj)
        Aj = construye_Ai(base_Aj, T)
        if j == 0:
            A = Aj
        else:
            A = np.vstack((A, Aj))
    return A

def construir_objPts():
    """Construye las coordenadas físicas de las 77 esquinas internas del tablero."""
    puntos = np.zeros((77, 2))
    k = 0
    for j in range(7):
        for i in range(11):
            puntos[k, 0] = 30 * i  # Coordenada U
            puntos[k, 1] = 30 * j  # Coordenada V
            k += 1
    return puntos

def resolverDLT(Q):
    A = matriz_x_matriz(transpuesta(Q), Q)
    S, D = diagRH(A)

    indice = 0
    for i in range(1, D.shape[0]):
        if D[i,i] < D[indice,indice]:
            indice = i

    m = S[:, indice]
    return m / norma(m, 2)

def resolverQhb(Q): #estamos duplicando código
    h = resolverDLT(Q)
    H = h.reshape(3, 3)
    
    return H

def construir_v(x, y):
    v=np.array([
        x[0] * y[0],
        x[0] * y[1] + x[1] * y[0],
        x[1] * y[1],
        x[0] * y[2] + x[2] * y[0],
        x[1] * y[2] + x[2] * y[1],
        x[2] * y[2]
    ])
    return v

def armarA(Hlst):
    for j in range(len(Hlst)):
        H = Hlst[j]
        h1 = H[:, 0]
        h2 = H[:, 1]
        fila1 = construir_v(h1, h2)
        fila2 = construir_v(h1, h1) - construir_v(h2, h2)
        Aj = np.vstack((fila1, fila2))
        if j == 0:
            A = Aj
        else:
            A = np.vstack((A, Aj))
    return A

def calcularMatrizK2D(A):
    b = resolverDLT(A)
    # Construyo B simétrica
    B = np.array([
    [b[0], b[1], b[3]],
    [b[1], b[2], b[4]],
    [b[3], b[4], b[5]]])
    if B[0, 0] < 0:
        B = -B
    if not(esSDP(B)):
        print("B no es definida positiva")
    L = calculaCholesky(B)
    R = L.T
    K = inversa(R)
    K = K / K[2, 2]
    return K



def producto_vectorial(u, v):
    w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w


def obtenerPose2D(Hj, K):
    M = matriz_x_matriz(inversa(K) , Hj)
    a1 = M[:, 0]
    a2 = M[:, 1]
    a3 = M[:, 2]
    lam = 1 / norma(a1,2)  
    if (lam * a3)[2] < 0:
        lam = -lam
    r1 = lam * a1
    r2 = lam * a2
    r3 = producto_vectorial(r1, r2) 
    R = np.hstack((r1.reshape(3, 1), r2.reshape(3, 1), r3.reshape(3, 1)))
    t = lam * a3
    return R, t


# Ejercicio 4

def obtenerQ2d_ejer4(imgPts, objPts):
    """Ecuación 11 (puntos coplanares): Q de 2n x 9."""
    return armar_Q(imgPts, objPts)

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


def estimar_homografia(img_pts, obj_pts):
    """Receta completa: normalizar, resolver, deshacer la normalización."""
    img_n, T_img = normalizar_puntos(img_pts)
    obj_n, T_obj = normalizar_puntos(obj_pts)
    H_n = resolver_H(obtenerQ2d_ejer4(img_n, obj_n))
    H = multiplicar_matrices(inversa(T_img), multiplicar_matrices(H_n, T_obj))
    for i in range(3):                        # dejar H[2,2] = 1 (solo prolijidad)
        for j in range(3):
            if (i, j) != (2, 2):
                H[i, j] = H[i, j] / H[2, 2]
    H[2, 2] = 1.0
    return H

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

def resolver_H(Q):
    """Ejercicio 3/4: devuelve H (3x3) = vector singular de Q asociado al menor valor singular."""
    return autovector_menor(Q, 3, 3)

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



"""MÓDULO 1"""
# =============================================================================
# Recibe dos numeros x e y, y calcula el error de aproximar 
# x usando y en float64
# =============================================================================
def error(x,y):
    x = np.float64(x)
    y = np.float64(y)
    return abs(y-x)
    
# =============================================================================
# Recibe dos numeros x e y, y calcula el error relativo de aproximar 
# x usando y en float64
# =============================================================================
def error_relativo(x,y):
    return error(x,y) / abs(x)

# =============================================================================
# Devuelve True si ambas matrices son iguales y False en otro caso.
# Considerar que las matrices pueden tener distintas dimensiones,
# ademas de distintos valores.
# =============================================================================

def matricesIguales(A,B):
    tol = 1e-08
    if A.shape != B.shape:
        return False
    col_A=len(A[0])
    filas_A=len(A)
    C = abs(B - A)
    for x in range(0,filas_A,1):
        for y in range(0,col_A,1):
            if C[x][y]>tol:
              return False
    return True



def rota(theta):
    a =np.cos(theta)
    b = -np.sin(theta)
    c = np.sin(theta)
    diagRH = np.cos(theta)
    M= np.array([[a,b],[c,diagRH]])
    return M

def graficar_vector(v):
    x = v[0]
    y = v[1]
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot([0, x], [0, y])
    ax.scatter(x, y)
    ax.axis('equal')

    plt.show()


def escala(s):
    n = len(s)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i][i] =s[i]
    return mat


def matriz_por_vector(A, v):
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
    res=np.zeros((m,j))

    for x in range(0,j,1):
        elem=matriz_por_vector(A,B[:,x])
        res[:,x]=elem
    return res

def rota_y_escala(theta,s):
    R =rota(theta)
    E =escala(s)
    res = matriz_x_matriz(E,R)
    return res

def afin(theta,s,b):
    mat = rota_y_escala(theta,s)
    res = np.zeros((3,3))
    for i in range(2):
        for j in range(2):
            res[i][j] = mat[i][j]
    res[0][2] = b[0]
    res[1][2] = b[1]
    res[2][2] = 1
    return res

def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)
    n, m = A.shape
    v3 = np.append(v,1)
    w = np.zeros(n-1) 
    for fila in range(n-1):
        suma = 0
        for columna in range(m):
            suma = suma + A[fila,columna]*v3[columna]
        w[fila] = suma
    return w

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


"""
def norma2(x, p):
   e
   #Calcula la norma p del vector x sin np.linalg.norm
    
    x = np.array(x)
    
    # 1. Norma 1
    if p == 1:
        return np.sum(np.abs(x))

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

    # 4. Cualquier otra norma p general (p=2, p=3, etc.)
    return np.sum(np.abs(x) ** p) ** (1 / p)
    """

def normaliza(x,p):
    lista = []
    for i in range(len(x)):
        if norma(x[i],p) != 0:
            lista.append(x[i]/norma(x[i],p))
    return lista

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

#si tomo todos los vectores x cuya longitud en norma p es exactamente 1,
#  ¿cuál es la longitud máxima (medida con la norma q) que puede alcanzar 
# el vector transformado Ax?
def normaMatMC(A, q, p, Np): #usamos.dot para que corra más rapido los tests, en los módulos siguientes sí usamos nuestra propia implementacion de multiplicación de matrices"""
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza el maximo.
    Esta versión es para pasar las pruebas de laboratorio.
    """
    # A = [a11,a12,...., a1m] es de tamaño n*m
    #     [a21,a22,...., a2m]
    #     [an1,an2,...., anm]

    vectores_aleatorios = np.random.standard_normal(size=(Np, A.shape[1]))
    # Genera vectores aleatorios en una matriz de np*#columnas de A 
    #   [b11,b12,...., b1m]                       np*m
    #   [b21,b22,...., b2m]
    
    # Normaliza con la norma 'p' 
    vectores_normalizados = normaliza(vectores_aleatorios, p) # va a tener misma dimension que vectores_aleatorios
    
    max_norma = 0
    max_vector = None
    # vectores normalizados:  [c11,c12,...., c1m]  es de np*m
    #                         [c21,c22,...., c2m]
    for x in vectores_normalizados:
        # Aplica la matriz A y calcula la norma de salida 'q'
        norma_actual = norma(matriz_por_vector(A, x), q)#vector_resultante= (n*m)@(?,1) donde ?=m 
        if norma_actual > max_norma:         #vector resultante va a ser de n*1
            max_norma = norma_actual
            max_vector = x

    return max_norma, max_vector

def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2. (c).
    """
    lista = []
    if p == 1 :
        A = transpuesta(A)
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    elif p=="inf" :
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    else :
        return

def maxNorma(A,p): # calcula el la norma 1 o infinito de una matriz
    n,m=A.shape
    max=0
    if p==1:
        B=transpuesta(A) # calculo max por columnas al transponerlo
        for fila in range(m):
            valor=norma(B[fila],1) # 1 porque quiero la suma de la fila para luego ver el max de las filas
            if valor >= max:
                max=valor
    else: # cuando p es inf
        for fila in range(n): # calculo el max por filas
            valor=norma(A[fila],1) # lo mismo, calculo las suma de la fila
            if valor>=max:
                max=valor
    return max

def normaExacta(A, p=[1, 'inf']):
    if p==1 or p=='inf':
        return maxNorma(A,p)
    elif p==[1, 'inf']:
        lista=[]
        lista.append(maxNorma(A,1))
        lista.append(maxNorma(A,'inf'))
        return lista
    else:
        return None
    
#version que pasa los tests
def condMC(A, p, Np):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    normaPdeA = normaMatMC(A,p,p,Np)[0]
    normaPdeAinv = normaMatMC(np.linalg.inv(A),p,p,Np)[0] #Estamos usando la función inversa de numpy porque es previo a calcular inversa con LU
    nroDeCondiconDeA = normaPdeA * normaPdeAinv
    return nroDeCondiconDeA
#version que no pasa los tests

def condMC2(A,p):
    return normaMatMC(A,p,p,1000)[0]*normaMatMC(np.linalg.inv(A),p, p,1000)[0]
          
def condExacta(A, p):
    """
    Que devuelva el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    normaPdeA = normaExacta(A, p)
    normaPdeAinv = normaExacta(np.linalg.inv(A), p)
    nroDeCondicionDeA = normaPdeA * normaPdeAinv
    return nroDeCondicionDeA

def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res

    
def multiplicar_matrices(A, B):
    """""
    A es m x p y B es p x n.
    """
    m, p1 = A.shape
    p2, n = B.shape
    if p1 != p2:
        return None
    C = np.zeros((m, n))
    # Bucle i: filas de A (m)
    for i in range(m):
        # Bucle j: columnas de B (n)
        for j in range(n):
            suma = 0
    # Bucle k: Itera sobre los elementos de la fila i de A y la columna j de B (dimensión p)
            for k in range(p1):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma      
    return C

#copiar la otra version!!
# --------------------------------------------------------------------
# calculaLU
# ---------------------------------------------------------------------
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

 
# ---------------------------------------------------------------------
# res_tri
# ---------------------------------------------------------------------
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

def detMatrizTriangular(U, tol=1e-10):  #funcion auxiliar para ver si U es sing
    determinante = 1 
    for i in range (U.shape[0]):
        determinante = determinante*U[i,i]
    if abs(determinante) < tol : 
        return 0 
    return determinante

# ---------------------------------------------------------------------
# inversa
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# calculaLDV
# ---------------------------------------------------------------------
def calculaLDV(A):
    """
    Calcula la factorización LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    """

    L,U,cant=calculaLU(A)
    if L is None or U is None:
        return None,None,None
    U_t=(transpuesta(U))
    V_t,D,c=calculaLU(U_t)
    V=np.array(transpuesta(V_t))

    return L,D,V
   
def esSDP(A, atol=1e-16): # cambia la diferencia a 16,por que los profes me dijeron que si se podia 
    """
    Checkea si la matriz A es simétrica definida positiva (SDP) usando
    la factorización LDV.
    """
    res_L,res_D,res_V=calculaLDV(A)
    if res_L is None:
        return False
    L, D, V = res_L,res_D,res_V
    n,m = L.shape 
    f,c = D.shape
    #1)veo si V=L^t
    for i in range(n):
        for j in range(m):
            if abs(A[i,j]-A[j,i]) > atol:
                return False
    #2)veo que todos los elementos de la diagonal sean > 0
    for k in range(f):
        if D[k,k]<=atol:
            return False
    return True

# ---------------------------------------------------------------------
# calculaChoesky
# ---------------------------------------------------------------------

def calculaCholesky(A, atol=1e-10):
    # Chequeo si es o no SDP
    if not esSDP(A):
        return None

    # Descomposición LU 
    L, U, c = calculaLU(A)
    n, m = U.shape
    
    # Construcción de D
    D = np.zeros((n, m))
    for i in range(n):
        diagRH = U[i, i]
        # El pivote debe ser estrictamente positivo
        if diagRH >= atol:
            D[i, i] = np.sqrt(diagRH)
        else:
            return None
        
    # Calculo R tal que A = R @ R.T (R = L @ D)
    R = multiplicar_matrices(L, D)
    return R



"""MODULO 5"""

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    A = np.asarray(A, "float64")
    p,m = A.shape
    
    if p != m: # Si A no es cuadrada, la hacemos cuadrada agregando filas o columnas de 0s
        if m > p:
            z = np.zeros((m-p,m))
            A = np.concatenate((A,z))
        else:
            z = np.zeros((p,p-m))
            A = np.concatenate((A,z),axis=1)


    n= A.shape[0] # Redefinimos a n y m
    # Creo matriz cuadrada de ceros para Q y R
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    Qprima = np.zeros((n,n)) # Matriz Q antes de normalizar las columnas
    nops = 0 # Contador de operaciones

    if not norma(A[:,0],2) < tol:

        Q[:,0] = A[:,0] / norma(A[:,0],2) # Primera columna de Q
        R[0,0] = norma(A[:,0],2) # Primer elemento de R
    
    
    for j in range(1,n): # Recorro columnas de A (y Q y R)
        Qprima[:,j] = A[:,j] # Copia cada columna de A a Qprima
        nops += n # Sumo 1 al numero de operaciones por copiar una columna

        for i in range(0,j): # Recorro columnas anteriores de Q
            R[i,j] = sum(Q[:,i] * Qprima[:,j]) # Calculo R[i,j]
            nops += 2*n - 1 # Operaciones del producto escalar
            Qprima[:,j] = Qprima[:,j] - R[i,j] * Q[:,i] # Actualizo Qprima
            nops += 2*n # Operaciones de la resta y multiplicacion por escalar
        
        R[j,j] = norma(Qprima[:,j],2) # Calculo R[j,j]
        nops += 2*n - 1 # Operaciones de la norma
        Q[:,j] = Qprima[:,j] / R[j,j] if R[j,j] > tol else 0 # Normalizo Qprima para obtener Q
        nops += n # Operaciones de la division por escalar
    
    if p > m:
        Q = Q[:,:m] # Reducimos Q
        R = R[:m,:m] # Reducimos R
    
    if p < m:
        Q = Q[:p,:] # Reducimos Q
        R = R[:p,:p] # Reducimos R

    if retorna_nops:
        return Q, R, nops
    
    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    (m, n) = np.shape(A)
    if m < n:
        return None
    R = np.copy(A)  #Inicializo la matriz R como copia de A
    Q = np.eye(m)   #Inicializo Q como matriz identidad de rango m 
    for k in range(n): #para cada columna:
        x = np.copy(R[k:,k]) #toma el vector columna de R de la posición k, comenzando por la fila k. Lo llama x
        x = np.array([x])                                   
        x = transpuesta(x) #traspone x
        a = - np.sign(x[0]) * norma(x,2) #construyo a
        e = np.zeros((m-k,1)) #construyo un vector canonico vertical de tamaño m-k
        e[0][0] = 1                                         
        u = x - (a*e) #construyo u. u genera un proyector que anula los elementos bajo la diagonal de la matriz A en la columna k
        if norma(u,2) > tol: #verifico que la norma 2 de u es mayor a la tolerancia
            u = u/(norma(u,2)) #normalizo u
            H = np.eye(m-k)                                 
            H2 = np.eye(m) #creo una matriz H2 identidad que contendrá a la matriz de householder
            H = H - 2*matriz_x_matriz(u,u.T) # crea la matriz de householder
            H2[k:,k:] = H[:,:] # inserto los valores de la matriz de householder en H2
            R = matriz_x_matriz(H2,R) # multiplica H2 por R para anular los elementos bajo la diagonal de R
            Q = matriz_x_matriz(Q,H2.T) # multiplica a Q por H2. Q será el producto de todas las matrices de householder generadas.

    R = R[:n,:]       #recortamos Q y R para que salga un R cuadrado y un Q con la misma cantidad de columnas que las filas de R
    Q = Q[:,:n]
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n n elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    metodo = metodo.upper() #MAYUSCULAS
    
    if metodo == 'RH':
        # Retorna Q y R calculadas con Householder
        # La función QR_con_HH no tiene un contador de operaciones
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    
    else:
        # Si el método no es 'RH' ni 'GS'
        return None
    

"""Auxiliares de estos ejercicios"""


def suma_vectorial(x):
    """
    Calcula la suma de todos los elementos del vector x
    """
    x = np.array(x) 
    acumulador = 0.0

    for elemento in x:
        acumulador += elemento
        
    return acumulador


"""MÓDULO 6"""

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

def diagRH(A, tol=1e-15,K=1000):
    if not(esSimetrica(A)):
        return None
    n = A.shape[0]
    v1,l1, _ = metpot2k(A,tol,K)
    #print(l1)
    #e1 = np.zeros(n)
    e1=vector_canonico(n,0)
    v1=np.asarray(v1, dtype=float).ravel()
    u=np.asarray(e1-v1)


    u_col=u.reshape(-1, 1)
    u_fila=u.reshape(1, -1)

    u_normalizado=norma(u,2)
    #H_v1 = np.eye(n) - 2*(np.outer(u,u)/ (u_normalizado**2))
    H_v1 = np.eye(n) - 2*(matriz_x_matriz(u_col,u_fila)/ (u_normalizado**2))
    #cambiar T por transpuesta manual
    if n == 2:
        S = H_v1
        D = matriz_x_matriz(H_v1,matriz_x_matriz(A,H_v1.T))
    else:
        B = matriz_x_matriz(H_v1,matriz_x_matriz(A,H_v1.T))
        AA = B[1:n,1:n]
        AA = (AA + AA.T) / 2 ## Corregimos la asimetría producida por redondeo
        SS, DD = diagRH(AA,tol,K)
        D = np.zeros((n,n))
        D[0][0] = l1.item()
        D[1:n,1:n] = DD
        SSS = np.zeros((n,n))
        SSS[0][0] = 1
        SSS[1:n,1:n] = SS
        S = matriz_x_matriz(H_v1,SSS)
    # Ajusto los elementos que no son diagonales
    for x in range(0,len(D),1):
        for y in range(0,len(D),1):
            if y!=x:
                D[x][y]=0.0

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


"""MÓDULO 7"""
def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    A = np.random.rand(n,n)
    for columna in range(n):
        A[::,columna] = A[::,columna] / norma(A[::,columna],1)
    return A


def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """

    A = np.random.rand(n,n)
    for columna in range(n):
        for fila in range(n):
            if (A[fila,columna] <= thres):
                A[fila,columna] = 1
            else:
                A[fila,columna] = 0
        if norma(A[::,columna],1) != 0 :
            A[::,columna] = A[::,columna] / norma(A[::,columna],1)
        else:
            A[np.random.randint(0,n),columna] = 1
    return A


def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz transpuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    B = matriz_x_matriz(transpuesta(A),A)
    S , D = diagRH(B)
    n =D.shape[0]
    nuc=[]
    if S is None or D is None:
        return None
    
    for i in range(n):
        autovalor = abs(D[i,i])
        if autovalor <= tol:
            nuc.append(i)
    # sin vectores no nulos en el nucleo
    tam=len(nuc) 
    if tam==0:
        return np.zeros((n,0))
    res= np.zeros((n,tam))
    col =0
    for i in nuc:
        res[:, col] = S[:, i]
        col +=1
    return res

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    if len(listado) == 0:
        return [{}, (m_filas, n_columnas)]
    
    diccionario = {}
    for i in range(len(listado[0])):
        if abs(listado[2][i]) >= tol:
            diccionario[(listado[0][i], listado[1][i])] = listado[2][i]
    
    return [diccionario, (m_filas, n_columnas)]

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    matrix_A = np.zeros((A[1][0], A[1][1]))
    for (i, j), value in A[0].items():
        matrix_A[i, j] = value

    w = matriz_por_vector(matrix_A, v)

    return w

"""Auxiliares de estos ejercicios"""

def transpuesta(A): 
    n, m = A.shape
    At = np.zeros((m, n))  # matriz de retorno donde ire cambiando valores
    for columna in range(m):
        for fila in range(n):
            At[columna, fila] = A[fila, columna]
    return At

        
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

"""MODULO 8"""
# los test de svd_reducida segun cuando consultamos nos dijeron que en el calculo de hat_U de numpy no devuelve respectivas matrices reducidas, devuelve el hat_U de la svd completa, por ende los test no coinciden con los calculados de nuestra funcion
# Y tambien observamos que en matrices con grandes nucleos por ejemplo de 12x4 si nos fijabamos directamente en los autovalores de A*A sin sacar raiz cuadrada de estos antes de comparar si son menores a la tolerancia, coincidian con los que devuelve numpy, es por eso que solo hicimos esa modificacion en el if y no en el else de nuestra funcion
def svd_reducida(A,k="max",tol=1e-15):
    """
    A: la matriz de interes (de m x n)
    k: el numero de valores singulares (y vectores) a retener.
    tol: la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) 
    y hatV (matriz de n x k)
    """
    m, n = A.shape
    k_valoresSingulares=[]
    #1) caso donde hay mas filas que columnas, uso la matriz A*A
    if m>=n: # calculo V de A*A
        C = matriz_x_matriz(transpuesta(A),A)
        V, D = diagRH(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = abs(autovalor)
            if (sigma>tol):
                k_valoresSingulares.append(np.sqrt(sigma)) # modificacion para que pase el test de 12X4
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        #Armo hatU usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatV = V[:, :c] # tomo las primeras c columnas
        hatU= np.zeros((m,c))
        B = multiplicar_matrices(A, hatV)
        for i in range(c):
            hatU[:,i] = B[:,i]/hatSig[i]
            hatU[0:m,i] = hatU[0:m,i]/norma((hatU[0:m,i]),2)
    # caso m<n mas columnas que filas
    # uso AA*
    else:
        C = matriz_x_matriz(A,transpuesta(A))
        U, D = diagRH(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = np.sqrt(abs(autovalor))
            if (sigma>tol):
                k_valoresSingulares.append(sigma)
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        #Armo hatV usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatU = U[:, :c]
        hatV= np.zeros((n,c))
        B = matriz_x_matriz(transpuesta(A), hatU)
        for i in range(c):
            hatV[:,i] = B[:,i]/hatSig[i]
            hatV[0:n,i] = hatV[0:n,i]/norma((hatV[0:n,i]),2)
    return hatU, hatSig, hatV


if __name__ == "__main__":
    detected_corners=' calib/3d/detected_corners.txt'
    p_w_corners=' calib/3d/p_W_corners.txt'
    datos_r2=np.loadtxt(detected_corners)
    datos_r3=np.loadtxt(p_w_corners,delimiter=',')    
    img_1_r2=datos_r2[0] #pruebo con la primer imagen
    Q=obtenerQ3d(img_1_r2,datos_r3)
    print(Q)

