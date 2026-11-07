import numpy as np
import os
from scipy.linalg import null_space

def agregar_elemento(v):
    """Incrementa en 1 la dimensión del vector, con 1 en la última posición"""
    w = np.ones(len(v) + 1)
    w[:-1] = v
    return w

def ordenar_factores(M):
    p=np.array([[0.,0.,1],[-1,0.,0.],[0.,-1,0.]])
    M=p@M
    return M

def construye_T():
    "Construye la matriz T(24x12) para obtener las ecuaciones de u y v"
    I = np.eye(4)
    O = np.zeros((4, 4))
    T = np.vstack([np.hstack([I, O, O]),
        np.hstack([O, O, O]),
        np.hstack([O, I, O]),
        np.hstack([O, O, O]),
        np.hstack([I, O, O]),
        np.hstack([O, O, I])])
    return T

def construye_Ai(M,T):
    """construye las matrices de la forma [[U V W 1 0 0 0 0 -Uu -Vu -Wu -u][0 0 0 0 U V W 1 -Uv -Vv -Wv -v]]"""
    z = M.reshape(12, 1)
    A_i = (T @ z).reshape(2, 12)
    return A_i

def obtenerQ3d(datos_2d_imagen, datos_3d):
    """"Recibe un elemento 2d y todo el dataset 3D"""
    T=construye_T()
    for j in range(0,len(datos_3d),1):
        coordenadas_3d=datos_3d[j]
        coordenadas_3d=agregar_elemento(coordenadas_3d)
        coordenadas_3d=coordenadas_3d.reshape(1, -1)
        par_uv=datos_2d_imagen[j*2:j*2+2] #elijo el par
        par_uv=agregar_elemento(par_uv)
        par_uv=par_uv.reshape(-1,1)
        base_Aj=par_uv@(coordenadas_3d)
        base_Aj=ordenar_factores(base_Aj)
        Aj=construye_Ai(base_Aj,T)
        if j==0:
            A=Aj
        else:
            A=np.vstack((A, Aj)) 
    return A


detected_corners= ' calib/3d/detected_corners.txt'
p_w_corners=' calib/3d/p_W_corners.txt'
datos_r2=np.loadtxt(detected_corners)
datos_r3=np.loadtxt(p_w_corners,delimiter=',')


# pi=(ui,vi) <-> Pi(Ui,Vi,Wi)
# Por cada imagen voy a 24 ecuaciones

#pruebo con la primer imagen
img_1_r2=datos_r2[0]
#selecciono el primer par u,v
Q=obtenerQ3d(img_1_r2,datos_r3)
# Recorro los elementos de 2D y busco la imagen con dim(nucleo) =1
print(datos_r2.shape)
for x in range(0,len(datos_r2),1):
    #print(datos_r2[x])
    Q=obtenerQ3d(datos_r2[x],datos_r3)
    nucleo = null_space(Q)
    print(f'El nucleo de la matriz {x} es : {nucleo}')
 



# EJERCICIO 2c Y 2b
def devolverMatricesCalibracion3d(M,igualar_focales=False,skew_cero=False):
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
    det = determinante3x3(R0)

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
    # restricciones del ej 2d
    if igualar_focales or skew_cero:
        t, K = aplicarRestricciones(K, t, igualar_focales, skew_cero)

    return R, t, K