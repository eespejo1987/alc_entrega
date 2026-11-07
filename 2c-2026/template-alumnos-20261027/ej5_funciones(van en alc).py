"""
EJERCICIO 5 - Cálculo de velocidad real en la calle Salguero.
Funciones hechas a mano (bucles). Usa las del módulo alc.py: producto_matriz, traspuesta, diagonalizar,
inversa, estimar_homografia, aplicar_homografia, armar_M, proyectar_M, error_reproyeccion_3d.

CADENA COMPLETA:
  (a) varias vistas del piso -> una homografía por vista -> Zhang -> K (intrínsecos)
  (b) los 51 puntos del piso -> una homografía -> con K se abre en R y t -> se reproyecta
  (c) cada captura: píxel del piso -> H^-1 -> metros -> (distancia entre capturas) / (tiempo) = velocidad
"""
import numpy as np
from alc import *


# =====================================================================================
# PARTE A: INTRINSECOS K  (método de Zhang, ejercicio 3)
# =====================================================================================
def vij(H, i, j):
    """Vector v_ij tal que  c_i^t B c_j = v_ij^t b,  con c_i = columna i de H y
    b = (B11, B12, B22, B13, B23, B33) las 6 entradas independientes de B = K^-t K^-1.
    Es solo ÁLGEBRA: se expande el producto c_i^t B c_j y se agrupa por entrada de B."""
    hi = H[:, i]
    hj = H[:, j]
    v = np.zeros(6)
    v[0] = hi[0] * hj[0]                       # coeficiente de B11
    v[1] = hi[0] * hj[1] + hi[1] * hj[0]       # coeficiente de B12 (aparece dos veces por simetría)
    v[2] = hi[1] * hj[1]                       # coeficiente de B22
    v[3] = hi[2] * hj[0] + hi[0] * hj[2]       # coeficiente de B13
    v[4] = hi[2] * hj[1] + hi[1] * hj[2]       # coeficiente de B23
    v[5] = hi[2] * hj[2]                       # coeficiente de B33
    return v


def armarA(Hlst):
    """Matriz A (2N x 6) de Zhang. Cada vista (cada H) aporta dos filas, que salen de que r1 y r2 son
    ortonormales:   r1^t r2 = 0  ->  c1^t B c2 = 0         ->  fila  v12
                    |r1| = |r2|  ->  c1^t B c1 - c2^t B c2 = 0  ->  fila  v11 - v22
    Se normaliza cada H  para que ninguna vista pese más que otra."""
    N = len(Hlst)
    A = np.zeros((2 * N, 6))
    for k in range(N):
        H = Hlst[k]
        s = 0.0
        for i in range(3):
            for j in range(3):
                s = s + H[i, j] ** 2
        Hn = H / np.sqrt(s)
        v12 = vij(Hn, 0, 1)
        v11 = vij(Hn, 0, 0)
        v22 = vij(Hn, 1, 1)
        for c in range(6):
            A[2 * k, c] = v12[c]
            A[2 * k + 1, c] = v11[c] - v22[c]
    return A


def cholesky(B):
    """Factorización de Cholesky a mano: B = L L^t con L triangular inferior.
    Devuelve None si B no es definida positiva (señal de que los datos no determinan bien K)."""
    n = B.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = B[i, j]
            for k in range(j):
                s = s - L[i, k] * L[j, k]
            if i == j:
                if s <= 0:
                    return None
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def calcularMatrizK2D(A):
    """Resuelve A b = 0 (mínimos cuadrados homogéneos, igual que el DLT: autovector de menor autovalor de
    A^t A), arma B, y saca K con Cholesky:  B = L L^t,  B = K^-t K^-1  =>  L = K^-t  =>  K = (L^t)^-1."""
    AtA = producto_matriz(traspuesta(A), A)
    S, D = diagonalizar(AtA)
    idx = 0
    for i in range(1, D.shape[0]):
        if D[i, i] < D[idx, idx]:
            idx = i
    b = S[:, idx]
    B = np.zeros((3, 3))
    B[0, 0] = b[0]; B[0, 1] = b[1]; B[1, 0] = b[1]
    B[1, 1] = b[2]; B[0, 2] = b[3]; B[2, 0] = b[3]
    B[1, 2] = b[4]; B[2, 1] = b[4]; B[2, 2] = b[5]
    if B[0, 0] + B[1, 1] + B[2, 2] < 0:        # b está definido salvo signo: elegimos B definida positiva
        B = -B
    L = cholesky(B)
    if L is None:
        return None
    K = inversa(traspuesta(L))
    return K / K[2, 2]                          # K33 = 1


def estimar_focal(Hlst, cx, cy):
    """(cuando Zhang completo es inestable): se supone skew = 0, píxeles cuadrados
    (alfa_u = alfa_v = f) y centro óptico (cx, cy) conocido (centro de la imagen). Queda UNA incógnita, f.
    Si trasladamos el origen de píxeles al centro, K = diag(f, f, 1) y las dos condiciones de Zhang dan:
        orto:        f^2 = -(a1 . a2) / (h31 h32)
        igual norma: f^2 = (|a1|^2 - |a2|^2) / (h32^2 - h31^2)
    con a_i = (primeras dos componentes de la columna i de H centrada) y h3i = su tercera componente.
    Devuelve las dos estimaciones de f por cada vista (nan si f^2 < 0)."""
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    res = []
    for H in Hlst:
        Hc = producto_matriz(T, H)
        c1 = Hc[:, 0]; c2 = Hc[:, 1]
        f2_orto = -(c1[0] * c2[0] + c1[1] * c2[1]) / (c1[2] * c2[2])
        f2_norma = ((c1[0] ** 2 + c1[1] ** 2) - (c2[0] ** 2 + c2[1] ** 2)) / (c2[2] ** 2 - c1[2] ** 2)
        fo = np.sqrt(f2_orto) if f2_orto > 0 else float('nan')
        fn = np.sqrt(f2_norma) if f2_norma > 0 else float('nan')
        res.append((fo, fn))
    return res


# =====================================================================================
# PARTE B: POSE (R, t) A PARTIR DE H Y K, Y REPROYECCION
# =====================================================================================
def producto_vectorial(a, b):
    """a x b para vectores de 3 componentes (np.cross no está permitido)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def ortogonalizar(R):
    """Rotación ortogonal a partir de R (que con ruido no es exactamente una rotación):
    las columnas se vuelven ortonormales con Gram-Schmidt (QR_con_GS) y nos quedamos con Q.
    Parámetro: R (3x3). Devuelve Q (3x3) ortogonal."""
    Q, _ = QR_con_GS(R)
    return Q


def obtenerPose2D_ej5(H, K):
    """Pose de la vista a partir de H = K [r1 r2 t] (salvo escala) y K.
        K^-1 H = lambda^-1 [r1 r2 t]   ->   lambda = 1/|K^-1 c1|,  r1 = lambda K^-1 c1,  r2 = lambda K^-1 c2,
        r3 = r1 x r2,  t = lambda K^-1 c3.
    El signo de lambda se elige para que el plano quede DELANTE de la cámara (t_z > 0)."""
    Ki = inversa(K)
    c1 = matriz_por_vector(Ki, H[:, 0])
    c2 = matriz_por_vector(Ki, H[:, 1])
    c3 = matriz_por_vector(Ki, H[:, 2])
    lam = 1.0 / np.sqrt(producto_interno(c1, c1))
    if c3[2] < 0:
        lam = -lam
    r1 = lam * c1
    r2 = lam * c2
    r3 = producto_vectorial(r1, r2)
    t = lam * c3
    R = np.zeros((3, 3))
    for i in range(3):
        R[i, 0] = r1[i]; R[i, 1] = r2[i]; R[i, 2] = r3[i]
    return ortogonalizar(R), t

#misma idea que el ejercicio 3


def centro_camara(R, t):
    """Posición de la cámara en el mundo: C = -R^t t (para chequear que la altura sea razonable)."""
    return -matriz_por_vector(traspuesta(R), t)


# =====================================================================================
# PARTE C: VELOCIDADES
# =====================================================================================
def tiempo_desde_nombre(nombre):
    """'img_10_18_47_351.jpg' -> segundos = h*3600 + m*60 + s + ms/1000.
    SUPUESTO: el nombre es hora_min_seg_ms (confirmarlo con los nombres reales de las capturas)."""
    partes = nombre.replace('.jpg', '').replace('.png', '').split('_')
    h = int(partes[-4]); m = int(partes[-3]); s = int(partes[-2]); ms = int(partes[-1])
    return h * 3600.0 + m * 60.0 + s + ms / 1000.0


def pixel_a_metros(H, pixeles):
    """Pixeles del PISO (n x 2) -> metros (n x 2) con H^-1 (la misma idea del inciso a del ejercicio 4)."""
    return aplicar_homografia(inversa(H), pixeles)


def velocidades_consecutivas(tiempos, pos):
    """Velocidad entre cada par de capturas consecutivas: |P(k+1) - P(k)| / (t(k+1) - t(k)), en m/s."""
    n = len(tiempos)
    v = np.zeros(n - 1)
    for k in range(n - 1):
        dx = pos[k + 1, 0] - pos[k, 0]
        dy = pos[k + 1, 1] - pos[k, 1]
        v[k] = np.sqrt(dx * dx + dy * dy) / (tiempos[k + 1] - tiempos[k])
    return v


def pendiente(t, y):
    """Pendiente de la recta de mínimos cuadrados y = a + b t (fórmula cerrada, sin librerías)."""
    n = len(t)
    mt = 0.0; my = 0.0
    for i in range(n):
        mt = mt + t[i] / n; my = my + y[i] / n
    num = 0.0; den = 0.0
    for i in range(n):
        num = num + (t[i] - mt) * (y[i] - my)
        den = den + (t[i] - mt) ** 2
    return num / den


def velocidad_por_ajuste(tiempos, pos):
    """Velocidad media a lo largo de toda la secuencia, ajustando una recta a X(t) y a Y(t)
    (más robusta al ruido que usar solo dos capturas). Devuelve (vx, vy, rapidez) en m/s."""
    vx = pendiente(tiempos, pos[:, 0])
    vy = pendiente(tiempos, pos[:, 1])
    return vx, vy, np.sqrt(vx * vx + vy * vy)

