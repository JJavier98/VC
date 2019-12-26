"""
Ejer3
num max de iter
la distancia entre dos pixeles es menor que un epsilon
combinación de esas dos
"""

"""
Ejer2
Mostrar 100 matches aleatorios
BFMatcher(crosscheck = True) -> se puede usar
Dmatch, trainIdx, queryIdx
Los match los devuelve en orden segun los ha ido encontrando
EL match guarda el indice de la lista de keypoints 1 y el indice de la lista de keypoints 2
Tenemos que ordenar las listas de manera que los índices coincidan en ambas listas
Estas listas ordenadas son las que le podemos pasar a findHomografy()
findHomografy() -> no podemos pasar directamente los keypoints de ambas imágenes
WarpPerspective() -> pasar a mosaico

La primera homografia es traslación
Para componer el mosaico componer las dos homografias (multiplicación de matrices)
Las homografias se sacan entre imagenes consecutivas no hacia el mosaico.
"""
"""
Orientacion en opencv [0 - 360] grados
no radianes, no negativos
"""

# Bibliotecas
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

###############################################################################
# ------------------------- Funciones auxiliares ---------------------------- #
###############################################################################

def leer_imagen(path):
    im_color = cv2.imread(path)
    im_trabajar = cv2.imread(path, 0)

    # Devolvemos la imagen para cv y para plt
    return np.float32(im_color), np.float32(im_trabajar)


def plt_imshow(im, title = ''):
    # Normalizamos imágenes a color
    if len(im.shape) == 3:
        im[:,:,0] = (im[:,:,0] - np.min(im[:,:,0])) / (np.max(im[:,:,0]) - np.min(im[:,:,0]))*255
        im[:,:,1] = (im[:,:,1] - np.min(im[:,:,1])) / (np.max(im[:,:,1]) - np.min(im[:,:,1]))*255
        im[:,:,2] = (im[:,:,2] - np.min(im[:,:,2])) / (np.max(im[:,:,2]) - np.min(im[:,:,2]))*255
        
    # SI es en ByN normalizamos la imagen entera (solo tiene un canal)
    else:
        im[:,:] = (im[:,:] - np.min(im[:,:])) / (np.max(im[:,:]) - np.min(im[:,:]))*255

    # Pasamos la imagen a enteros sin signo [0,255]
    im = np.uint8(im)

    # Si la imagen es a color
    if len(im.shape) == 3:
        im_plt = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) # La cambiamos a RGB
        plt.imshow(im_plt) # mostramos
        
    else: # Si es gris
        plt.imshow(im, cmap='gray') # Mostramos en escala de grises
    plt.title(title) # añadimos un título

# ## Gaussiana 2D 
# 
# Obtenemos el kernel de una función Gaussiana de tamaño _kernel\_size_, $\sigma$ = _sigma_ y bordes de tipo _borde_. Tras esto multiplicamos el kernel obtenido por su traspuesta para obneter una máscara 2D para aplicarla a la imagen.
def Gaussiana2D(im, kernel_size, sigma, borde):

    # Obtenemos una máscara 1D de tamaño <kernel_size> de una función Gaussiana de sigma = <sigma>
    gk = cv2.getGaussianKernel(kernel_size,sigma)
    # Obtenemos la máscara 2D mediante la multiplicación matricial de la máscara
    # 1D por su transpuesta
    gk_2d = np.dot(gk, gk.T)
    # Aplicamos la máscara 2D
    im2 = cv2.filter2D(im, -1, gk_2d, borderType=borde)

    return im2

# ## Subsample
# 
# Reducimos la imagen pasada como parámetro suprimiendo las filas y columnas pares.
def imageSubsample(im):
    # Creamos una "imagen" (matriz) con unas dimensiones igual a la mitad (en ambos ejes) de la original
    im2 = np.ones((int(im.shape[0]/2), int(im.shape[1]/2)))
    
    # Copiamos la imagen original en la creada anteriormente
    # saltándonos las filas y columnas de 2 en 2
    for i in range(int(im.shape[0]/2)):
        for j in range(int(im.shape[1]/2)):
            im2[i,j] = im[i*2,j*2]
            
    return im2


# ## Upsample
# 
# Aumentamos el tamaño de la imagen pasada como argumento insertando cada fila y columna dos veces
def imageUpsample(im):
    # Creamos una "imagen" (matriz) con unas dimensiones igual al doble (en ambos ejes) de la original
    im2 = np.ones((int(im.shape[0]*2), int(im.shape[1]*2)))
    
    # Copiamos la imagen original en la creada anteriormente
    # doplicando las filas y columnas de 2 en 2
    for i in range(int(im2.shape[0])):
        for j in range(int(im2.shape[1])):
            im2[i,j] = im[int(i/2),int(j/2)]
            
    return im2

# ## Sobel (getDerivKernels)
# 
# Aplicamos Sobel de tamaño _kernel\_size_ en _x_ e _y_ a la imagen _im_ con configuración de bordes = _borde_ .  
# Sobel suaviza la imagen y después calcula su derivada en un determinado eje con la intención de detectar los bordes de la imagen o, más concretamente, los cambios de frecuencia (intensidad de color).  
# Obtenemos los kernels de ambos ejes y los multiplicamos matricialmente para obtener una máscara 2D la cual tendremos que voltear en ambos ejes para que al aplicarla realicemos una convolución. Tras esto aplicamos estas máscaras por separado obteniendo dos imágenes una con un resalto de bordes en el eje horizontal y otra en el vertical.
def gradiente(im, kernel_size, borde = cv2.BORDER_DEFAULT):

    kx = cv2.getDerivKernels(1, 0, kernel_size, normalize=True) # kernel para derivar + kernel para suavizar en x
    ky = cv2.getDerivKernels(0, 1, kernel_size, normalize=True) # kernel para suavizar + kernel para derivar en y

    kx = np.dot(kx[1], kx[0].T) # Obtenemos una máscara 2D para aplicar la máscara de Sobel en x (suavizado x derivada)
    ky = np.dot(ky[1], ky[0].T) # Obtenemos una máscara 2D para aplicar la máscara de Sobel en y (derivada x suavizado)

    # Hacemos un flip en ambos ejes en las dos máscaras
    # para realizar una convolución en vez de una correlación
    kx = cv2.flip(kx, -1)
    ky = cv2.flip(ky, -1)

    # Aplicamos la máscara Sobel en X y en Y
    im2 = cv2.filter2D(im, -1, kx, borderType=borde)
    im3 = cv2.filter2D(im, -1, ky, borderType=borde)
    
    # Devolvemos las imágenes con la máscara aplicada
    return im2, im3


def pyramid(v_img):
    # Construcción de una imagen que contiene los niveles de la pirámide Gaussiana
    # Vamos concatenando las distintas octavas para que se muestren juntas en una sola imagen
    base = np.ones((v_img[0].shape[0], v_img[0].shape[1] + v_img[1].shape[1]))
    if(len(v_img[0].shape) == 3):
        base = np.ones((v_img[0].shape[0], v_img[0].shape[1] + v_img[1].shape[1], 3))
    for i in range(v_img[0].shape[0]):
        for j in range(v_img[0].shape[1]):
            base[i][j] = v_img[0][i][j]

    offset_col = v_img[0].shape[1]
    
    for i in range(v_img[1].shape[0]):
        for j in range(v_img[1].shape[1]):
            base[i][j+offset_col] = v_img[1][i][j]

    offset_fil = v_img[1].shape[0]
    
    for i in range(v_img[2].shape[0]):
        for j in range(v_img[2].shape[1]):
            base[i+offset_fil][j+offset_col] = v_img[2][i][j]

    offset_fil += v_img[2].shape[0]
    
    for i in range(v_img[3].shape[0]):
        for j in range(v_img[3].shape[1]):
            base[i+offset_fil][j+offset_col] = v_img[3][i][j]

    offset_fil += v_img[3].shape[0]
    
    for i in range(v_img[4].shape[0]):
        for j in range(v_img[4].shape[1]):
            base[i+offset_fil][j+offset_col] = v_img[4][i][j]
    
    # Devolvemos la pirámide
    return base

def GaussianPyramid(im, kernel_size = 7, sigma = 3):
    im2 = np.copy(im)
    v_imgs = []

    v_imgs.append(im2) # level 0 - original
    im2 = cv2.pyrDown(im2) # Downsampling
    # Cuatro niveles
    for i in range(4):
        im2 = Gaussiana2D(im2, kernel_size, sigma, cv2.BORDER_DEFAULT) # Filtro de Gaussiana
        v_imgs.append(im2)
        im2 = cv2.pyrDown(im2) # Downsampling

    return v_imgs


###############################################################################
# ------------------------- Funciones auxiliares ---------------------------- #
###############################################################################

def supNoMax(R):
    """
    Realizamos la supresión de no máximos de la imagen R
    y almacenamos el resultado en no_max
    level_indices almacena los índices de los puntos máximos
    """
    no_max = np.float32(np.zeros_like(R))
    level_indices = []
    for k in range(R.shape[0]):
        for j in range(R.shape[1]):
            maxi = True
            main = R[k][j]
            
            if (k-1) >= 0:
                if R[k-1][j] >= main:
                    maxi = False
                if (j-1) >= 0:
                    if R[k-1][j-1] >= main:
                        maxi = False
                if (j+1) < R.shape[1]:
                    if R[k-1][j+1] >= main:
                        maxi = False
                        
            if (k+1) < R.shape[0]:
                if R[k+1][j] >= main:
                    maxi = False
                if (j-1) >= 0:
                    if R[k+1][j-1] >= main:
                        maxi = False
                if (j+1) < R.shape[1]:
                    if R[k+1][j+1] >= main:
                        maxi = False
                        
            if (j-1) >= 0:
                if R[k][j-1] >= main:
                    maxi = False
                        
            if (j+1) < R.shape[1]:
                if R[k][j+1] >= main:
                    maxi = False
            
            # Si es máximo lo guardamos
            if maxi:
                no_max[k][j] = main
                level_indices.append((k,j))

    return no_max, level_indices
    

def calculaKeyPoints(path):
    """
    Calculamos los puntos de Harris para una imagen dada
    en diferentes octavas
    """
    # Tamaño del bloque para calcular los puntos de Harris
    BlockSize = 7
    #Lectura de imagen
    im_color, im_tr = leer_imagen(path)
    # Cálculo de niveles de la pirámide Gaussiana
    v_pyr = GaussianPyramid(im_tr)
    v_pyr_color = GaussianPyramid(im_color)
    # Declaramos un umbral suficiente para obtener
    # algo más de 2000 puntos en total
    umbral = 90
    # Guardamos las imágenes con la supresión de los no máximos
    v_no_max = []
    # Guardamos los índices en la que hay máximos locales en cada nivel
    indices = []
    # Calculamos gradientes de la imagen original
    grad_x, grad_y = gradiente(im_tr, 5)
    # Aplicamos un suavizado con sigma = 4.5 a ambos gradientes
    grad_x = Gaussiana2D(grad_x, 5, 4.5, cv2.BORDER_DEFAULT)
    grad_y = Gaussiana2D(grad_y, 5, 4.5, cv2.BORDER_DEFAULT)
    # Calculamos la pirámide Gaussiana para cada gradiente
    pyr_grad_x = GaussianPyramid(grad_x)
    pyr_grad_y = GaussianPyramid(grad_y)
    # Vector de niveles con coordenadas reales
    v_kp_junto_relativas = []
    # Vectores con todos los keypoints
    keypoints = []
    # Almacena por niveles los puntos Harris con umbral sin supresion de no máximos
    v_corner = []
    for i in range(len(v_pyr)):
        # Detección de esquinas
        eigs = cv2.cornerEigenValsAndVecs(v_pyr[i], BlockSize, 7)
        # Nos quedamos con el valor de las lambdas
        eigs = eigs[:, :, 0:2]
        Lambda1 = eigs[:, :, 0]
        Lambda2 = eigs[:, :, 1]

        # Calcualmos el operador Harris
        R = (Lambda1 * Lambda2)/(Lambda1 + Lambda2)

        # Mostramos el resultado antes del umbral
        input('Mostrar puntos encontrados antes del umbral')
        plt_imshow(R, 'antes de umbral')
        plt.show()
        
        # Aplicamos el umbral
        for j in range(R.shape[0]):
            for k in range(R.shape[1]):
                if(R[j][k] < umbral):
                    R[j][k] = 0
        #Guardamos los resultados de cada nivel
        v_corner.append(R)

        # Contamos los puntos restantes
        inc = 0
        for j in range(R.shape[0]):
            for k in range(R.shape[1]):
                if(R[j][k] >= umbral):
                    inc += 1
        print('puntos post-umbral en nivel ' + str(i) + ': ' + str(inc))
        
        # Mostramos después de aplicar el umbral
        input('Mostrar puntos encontrados después del umbral')
        plt_imshow(R, 'después de umbral')
        plt.show()
        

        # Realizamos la supresión de no máximos en un vecindario 3x3      
        #for k in range(len(v_corner)):
        no_max, level_indices = supNoMax(R)

        # Contamos los puntos restantes tras supresión
        inc = 0
        for j in range(no_max.shape[0]):
            for k in range(no_max.shape[1]):
                if(no_max[j][k] > 0):
                    inc += 1
        print('puntos post-supresión en nivel ' + str(i) + ': ' + str(inc))

        # Normalizamos no_max
        no_max[:,:] = (no_max[:,:] - np.min(no_max[:,:])) / (np.max(no_max[:,:]) - np.min(no_max[:,:]))*255
        # Guardamos las imágenes con los puntos máximos por niveles
        v_no_max.append(no_max)
        # Guardamos los índices de los puntos máximos para evitar recorrer la matriz
        indices.append(level_indices)

        # Mostramos imágenes despues de suprimir no máximos
        input('Mostrar puntos encontrados después de supresión de no máximos')
        plt_imshow(v_no_max[i], 'despues de supresion')
        plt.show()
        
        # Coordenas reales guardadas en cada posición del nivel
        kp_junto_relativas = []
        # Escala real
        escala = BlockSize * (i+1)

        for j in range(len(level_indices)):
            # Obtenemos los índices
            a, b = level_indices[j]
            # Calculamos el módulo del vector
            modulo_U = np.sqrt(pyr_grad_x[i][a][b]*pyr_grad_x[i][a][b] + pyr_grad_y[i][a][b]*pyr_grad_y[i][a][b])
            # Calculamos cos y sin
            cos_x = pyr_grad_x[i][a][b] / modulo_U
            sin_x = pyr_grad_y[i][a][b] / modulo_U
            # Calculamos el ángulo (orientación del keypoint)
            # El resultado en radianes comprendidos en [-pi, +pi]
            # así qu elo pasamos a grados 180/pi
            angulo = np.arctan2(cos_x, sin_x)*180/np.pi
            # Si es negativo lo pasamos a positivo
            while(angulo < 0):
                angulo += 360

            # Como la imagen se trata como un eje cartesiano
            # (las filas son el eje y, y las columnas son el eje x)
            # tenemos que cambiar filas por columnas halladas
            fila_real = np.int64(a*(2**i)) # coordenada de las columnas
            columna_real = np.int64(b*(2**i)) # coordenada de las filas
            # Creamos el KeyPoint
            kp = cv2.KeyPoint(columna_real, fila_real, escala, angulo)
            # Guardamos el KeyPoint
            keypoints.append(kp)
            # Guardamos el KP junto a sus coordenadas relativas y el nivel donde se encuentra
            kp_junto_relativas.append([i, a, b, kp])

        # Guardamos el kp con datos relativos por niveles
        v_kp_junto_relativas.append(kp_junto_relativas)

    """
    APARTADO D - NO FUNCIONA
    # Creamos keypoint refinados
    # Definimos el criterio de parada
    # tras 30 iteraciones o con un error epsilon menor a 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0005)
    # Lista de Keypoints refinados con coordenadas relativas
    kp_ref_rel = []
    # Para cada nivel de la pirámide
    for i in range(len(v_kp_junto_relativas)):
        new_points = []
        # recorremos los keypoints de cada nivel
        kp_level_i = v_kp_junto_relativas[i]
        for kp in kp_level_i:
            if kp[0] == i:
                # Cogemos todos los keypoints de ese nivel para refinarlos
                new_points.append(kp[3].pt)
        # Transformamos el tipo de las coordenadas para refinarlas
        new_points = np.float32(new_points)
        # Obtenemos las coordenadas relativas refinadas
        # (5,5) hace referencia a (5*2+1, 5*2+1) el tamaño que tendrá la ventana
        # (-1,-1) indica que no habrá cambio en la zona muerta
        kp_ref_rel.append(cv2.cornerSubPix(v_pyr[i], new_points, (5,5), (-1,-1), criteria))

    # Igual que 'keypoints' pero con coordenadas refiandas
    # keypoints refinados reales
    kp_ref_real = []
    # keypoints refinados reales junto a sus coordenadas relativas
    kp_ref_real_junto_relativas = []
    # vector de keypoints refinados reales junto a sus coordenadas relativas por niveles de pirámide
    v_kp_ref_real_junto_relativas = []
    # Creamos los nuevos keypoints
    # Recorremos los niveles de la pirámide
    for i in range(len(v_pyr)):
        # Accedemos a los valores relativos refinados
        for j in range(len(kp_ref_rel[i])):
            # Ya no tenemos que intercambiar filas por columnas ya que
            # el refinamiento nos da las coordenadas en orden
            columna_real = np.int64(kp_ref_rel[i][j][1]*2**i)
            fila_real = np.int64(kp_ref_rel[i][j][0]*2**i)
            # obtenemos la escala y el ángulo de los keypoints sin refinar
            # estos datos no cambian
            escala = v_kp_junto_relativas[i][j][3].size
            angulo = v_kp_junto_relativas[i][j][3].angle
            # creamos los nuevos keypoints refinados y los guardamos
            kp_n = cv2.KeyPoint(fila_real, columna_real, escala, angulo)
            kp_ref_real.append(kp_n)
            # guardamos estos keypoints junto a informacion referente a el nivel de la pirámide
            # y sus coordenadas relativas
            kp_ref_real_junto_relativas.append([i, kp_ref_rel[i][j][0], kp_ref_rel[i][j][1], kp_n])
        # almacenamos los nuevos keypoints junto a los datos auxilares por niveles de pirámide
        v_kp_ref_real_junto_relativas.append(kp_ref_real_junto_relativas)


    kp_dif = []
    kp_dif_index = []
    for i in range(len(v_pyr)):
        for j in range(len(v_kp_junto_relativas)):
            if(v_kp_junto_relativas[i][j][3].pt[0] != v_kp_ref_real_junto_relativas[i][j][3].pt[0] or
            v_kp_junto_relativas[i][j][3].pt[1] != v_kp_ref_real_junto_relativas[i][j][3].pt[1]):
                kp_dif.append([v_kp_junto_relativas[i][j][3], v_kp_ref_real_junto_relativas[i][j][3]])
                kp_dif_index.append((i,j))
                print(v_kp_junto_relativas[i][j][1])
                print(v_kp_junto_relativas[i][j][2])
                print(v_kp_ref_real_junto_relativas[i][j][1])
                print(v_kp_ref_real_junto_relativas[i][j][2])
                print(v_kp_junto_relativas[i][j][3].pt)
                print('')
                
    refinados = []
    no_refinados = []
    for i in range(len(kp_dif)):
        refinados.append(kp_dif[i][1])
        no_refinados.append(kp_dif[i][0])
    """

    # Vamos a dibujar los Keypoints en los niveles donde fueron hallados
    for i in range(len(v_pyr_color)):
        local_kp = []
        # Recorremos los keypoints junto a sus valores relativos al nivel
        for j in range(len(v_kp_junto_relativas[i])):
            # Creamos el KP con las coordenadas relativas y lo almacenamos
            n_kp = cv2.KeyPoint(v_kp_junto_relativas[i][j][2], v_kp_junto_relativas[i][j][1], v_kp_junto_relativas[i][j][3].size, v_kp_junto_relativas[i][j][3].angle)
            local_kp.append(n_kp)

        # Seleccionamos en nivel de la pirámide donde vamos a pintar
        kp_image = v_pyr_color[i]
        # Pintamos los key points
        kp_image = cv2.drawKeypoints(np.uint8(v_pyr_color[i]), local_kp, v_pyr_color[i])
        text = 'Mostrar keypoints del nivel ' + str(i)
        input(text)
        plt_imshow(kp_image)
        plt.show()
    
    # Por último dibujamos todos los KP en el nivel 0 (imagen original)
    kp_image = im_color
    kp_image = cv2.drawKeypoints(np.uint8(im_color), keypoints, im_color)
    input('Mostrar TODOS los keypoints')
    plt_imshow(kp_image)
    plt.show()

###############################################################################
# ------------------------------ Ejercicios --------------------------------- #
###############################################################################

def ejer1():
    """
    Calcula los puntos de Harris en las dos imágenes de Yosemite.zip
    Yosemite1.jpg y Yosemite1.jpg
    Una vez calculados los muestra por pantalla por niveles de la pirámide
    Gaussiana calculada a la imagen y, finalmente, 
    muestra todos en la imagen original
    """
    calculaKeyPoints('imagenes/Yosemite1.jpg')
    calculaKeyPoints('imagenes/Yosemite2.jpg')

def ejer2():
    """
    Obtiene los desriptores AKAZE de las imagenes de yosemite
    y muestra por pantalla los matches entre los keypoints encontrados
    uniéndolos con líneas. Mostramos "sólo" 50 matches
    ya que más estorban a la hora de observar la precisión
    """
    # Declaramos el path de todas las imágenes
    imgs = ['imagenes/yosemite1.jpg', 'imagenes/yosemite2.jpg',
            'imagenes/yosemite3.jpg', 'imagenes/yosemite4.jpg',
            'imagenes/yosemite5.jpg', 'imagenes/yosemite6.jpg',
            'imagenes/yosemite7.jpg']

    # Redorremos todas las imágenes para calcular los descriptores AKAZE por pares
    for i in range(len(imgs)-1):
        # Leemos las imgs
        im_color_1, im_tr_1 = leer_imagen(imgs[i])
        im_color_2, im_tr_2 = leer_imagen(imgs[i+1])

        # Declaramos el descriptor
        akaze = cv2.AKAZE_create()
        # Calculamos los KeyPoints y los descriptores para cada imagen
        kp_1, desc1 = akaze.detectAndCompute(im_tr_1, None)
        kp_2, desc2 = akaze.detectAndCompute(im_tr_2, None)

        # Declaramos un Matcher de fuerza bruta (el más cercano)
        matcher = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE, crossCheck=True)
        # Realizamos el match
        matches1 = matcher.match(desc1, desc2)
        # Ordenamos los matches por distancia
        # Se emparejan con con el más cercano que no sea el mismo
        # 2NN
        matches1 = sorted(matches1, key = lambda x:x.distance)
        # Mezclamos los resultados para mostrar matches aleatorios
        random.shuffle(matches1)
        # Mostramos estos matches por pantalla
        img1 = cv2.drawMatches(np.uint8(im_color_1),kp_1,np.uint8(im_color_2),kp_2,matches1[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        input('Mostrar enlaces encontrados entre ambas imágenes')
        plt_imshow(img1)
        plt.show()

def ejer3(path1, path2):
    """
    Realizamos un mosaico de calidad con dos imágenes de 'mosaico'
    de las que obtenemos los KeyPoints igual que en el ejercicio 2
    """
    # Leemos las imáges de mosaico
    im_color_src, im_gray_src = leer_imagen(path1)
    im_color_dst, im_gray_dst = leer_imagen(path2)

    # Declaramos el descriptor
    akaze = cv2.AKAZE_create()
    # Calculamos los KeyPoints y los descriptores para cada imagen
    kp_src, desc1 = akaze.detectAndCompute(im_gray_src, None)
    kp_dst, desc2 = akaze.detectAndCompute(im_gray_dst, None)
    # Declaramos un Matcher de fuerza bruta (el más cercano)
    matcher = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE, crossCheck=True)
    # Realizamos el match
    matches1 = matcher.match(desc1, desc2)
    # Ordenamos los matches por distancia
    # Se emparejan con con el más cercano que no sea el mismo
    # 2NN
    matches1 = sorted(matches1, key = lambda x:x.distance)
    # Nos quedamos con tan solo los 30 primeros matches
    # ya que son aquellos que tienen más calidad y son suficientes
    # para calcular la homografía
    matches1 = matches1[:30]

    # Dibujamos en las imágenes del mosaico los matches encontrados (los 30 mejores)
    img1 = cv2.drawMatches(np.uint8(im_color_src),kp_src,np.uint8(im_color_dst),kp_dst,matches1,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Formateamos los keypoints para pasarselos a la función de la homografía
    src_points = np.zeros((len(matches1), 2), dtype=np.float32)
    dst_points = np.zeros((len(matches1), 2), dtype=np.float32)
    for i, match in enumerate(matches1):
        src_points[i, :] = kp_src[match.queryIdx].pt
        dst_points[i, :] = kp_dst[match.trainIdx].pt

    # Calculamos la matrix homográfica
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5)
    
    # Declaramos un tamaño para el canvas del mosaico
    size = (400, 250)
    # Calculamos el mosaico con warpPerspective
    # Transporta la imagen a encajar al mosaico
    mosaico = cv2.warpPerspective(im_color_src, H, size, borderMode=cv2.BORDER_TRANSPARENT)
    # La imagen sobre la que se encaja se define en el (0,0) del mosaico
    mosaico[0:im_color_dst.shape[0], 0:im_color_dst.shape[1]] = im_color_dst

    input('Mostrar matches entre imágenes')
    plt_imshow(img1)
    plt.show()

    #input('Mostrar mosaico')
    plt_imshow(mosaico)
    plt.show()

    # imagenes con matches, imágenes superpuestas
    return img1, mosaico

def ejer4():
    """
    Como en la anterior, debemos hacer un mosaico encajando imágenes sobre otras
    según nos indiquen sus KP y los matches que hay entre ellos
    Como en el ejer3 empezamos por la priemra imagen que se posiciona
    en la esquina superir izda
    """

    # Leemos todas las imágenes que formarán parte del mosaico
    im_color_1, im_gray_1 = leer_imagen('imagenes/mosaico002.jpg')
    im_color_2, im_gray_2 = leer_imagen('imagenes/mosaico003.jpg')
    im_color_3, im_gray_3 = leer_imagen('imagenes/mosaico004.jpg')
    im_color_4, im_gray_4 = leer_imagen('imagenes/mosaico005.jpg')
    im_color_5, im_gray_5 = leer_imagen('imagenes/mosaico006.jpg')
    im_color_6, im_gray_6 = leer_imagen('imagenes/mosaico007.jpg')
    im_color_7, im_gray_7 = leer_imagen('imagenes/mosaico008.jpg')
    im_color_8, im_gray_8 = leer_imagen('imagenes/mosaico009.jpg')
    im_color_9, im_gray_9 = leer_imagen('imagenes/mosaico010.jpg')
    im_color_10, im_gray_10 = leer_imagen('imagenes/mosaico011.jpg')

    # dividimos las imágenes por color y escala de grises
    ims_color = [im_color_1, im_color_2, im_color_3, im_color_4, im_color_5, im_color_6, im_color_7, im_color_8, im_color_9, im_color_10]
    ims_gray = [im_gray_1, im_gray_2, im_gray_3, im_gray_4, im_gray_5, im_gray_6, im_gray_7, im_gray_8, im_gray_9, im_gray_10]
    # definimos una lista de homografías
    H = []

    # Declaramos un tamaño para el canvas donde irá el mosaico
    size = (1100, 550)
    # inicializamos el mosaico a una imagen en negro
    mosaico = np.zeros(size, dtype=np.uint8)
    # Recorremos todas las imágenes por pares para calcular sus matches
    for i in range(len(ims_color)-1):
        # Declaranos el descriptor
        akaze = cv2.AKAZE_create()
        # Computamos los KeyPoints y los descriptores
        kp_src, desc1 = akaze.detectAndCompute(ims_gray[i+1], None)
        kp_dst, desc2 = akaze.detectAndCompute(ims_gray[i], None)
        # Declaramos el Matcher
        matcher = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE, crossCheck=True)
        # Calculamos los matches entre imágenes
        matches1 = matcher.match(desc1, desc2)
        # Emparejamos cada match con su correspondiente según cercanía
        matches1 = sorted(matches1, key = lambda x:x.distance)
        # Nos quedamos con los 100 mejores para realizar el mosaico
        matches1 = matches1[:100]

        # Dibujamos en las imágenes los matches encontrados
        img1 = cv2.drawMatches(np.uint8(ims_color[i+1]),kp_src,np.uint8(ims_color[i]),kp_dst,matches1[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        input('Mostrar matches entre imágenes')
        plt_imshow(img1)
        plt.show()

        # Formateamos los KeyPoints para pasárselos a la homgrafía
        src_points = np.zeros((len(matches1), 2), dtype=np.float32)
        dst_points = np.zeros((len(matches1), 2), dtype=np.float32)
        for i, match in enumerate(matches1):
            src_points[i, :] = kp_src[match.queryIdx].pt
            dst_points[i, :] = kp_dst[match.trainIdx].pt

        # Calcualmos la matriz de homografía
        h, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5)
        # La almacenamos
        H.append(h)
    
    # Cogemos la primera homografía
    current_H = H[0]
    for i in range(len(ims_color)-1):
        # Si no es la primera imagen
        if(i != 0):
            # Calculamos la homografía correspondiende como composición de la homografía
            # de la imagen anterior y la actual
            current_H = current_H@H[i]
            # Pintamos la imagen actual en el mosaico (canvas) según nos marca la homografía
            cv2.warpPerspective(ims_color[i+1], current_H, size, dst=mosaico, borderMode=cv2.BORDER_TRANSPARENT)
        # Si es la primera imagen
        else:
            # Calculamos el mosaico pintando la primera (segunda) imagen sobre él
            mosaico = cv2.warpPerspective(ims_color[i+1], current_H, size, dst=mosaico, borderMode=cv2.BORDER_TRANSPARENT)

    # la primera imagen (inicial) se pinta en el (0,0) del mosaico
    mosaico[0:ims_color[0].shape[0], 0:ims_color[0].shape[1]] = ims_color[0]

    input('Mostrar mosaico con Todas las imágenes')
    plt_imshow(mosaico)
    plt.show()


#ejer1()
#ejer2()
#ejer3('imagenes/mosaico003.jpg', 'imagenes/mosaico002.jpg')
ejer4()