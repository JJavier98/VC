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

def leer_imagen(path):
    im_cv = cv2.imread(path)

    # Devolvemos la imagen para cv y para plt
    return np.float64(im_cv)


def plt_imshow(im, title = ''):
    
    # NORMALIZAMOS  
    # Si es a color
    if len(im.shape) == 3:
        im[:,:,0] = (im[:,:,0] - np.min(im[:,:,0])) / (np.max(im[:,:,0]) - np.min(im[:,:,0]))*255
        im[:,:,1] = (im[:,:,1] - np.min(im[:,:,1])) / (np.max(im[:,:,1]) - np.min(im[:,:,1]))*255
        im[:,:,2] = (im[:,:,2] - np.min(im[:,:,2])) / (np.max(im[:,:,2]) - np.min(im[:,:,2]))*255
        
    # SI es en ByN normalizamos la imagen entera (solo tiene un canal)
    else:
        im[:,:] = (im[:,:] - np.min(im[:,:])) / (np.max(im[:,:]) - np.min(im[:,:]))*255
        
    plt.imshow(im) # Mostramos en escala de grises
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


def pyramid(v_img):
    # Creamos imágenes que serán el fondo de las imágenes de la pirámide
    blanco2 = np.ones((v_img[1].shape[0]-v_img[2].shape[0], v_img[2].shape[1]))*255 # Fondo para nivel 2 (mismo tamaño)
    blanco3 = np.ones((v_img[1].shape[0]-v_img[3].shape[0], v_img[3].shape[1]))*255
    blanco4 = np.ones((v_img[1].shape[0]-v_img[4].shape[0], v_img[4].shape[1]))*255

    blanco5 = np.ones((v_img[4].shape[0]*2, v_img[1].shape[1]-v_img[4].shape[1]))*255 # Fondo para nivel 4
    
    # Fijamos los distintos niveles de la pirámide
    level0 = v_img[0]
    level1 = v_img[1]
    # Concatenamos con los fondos para que tengan el tamaño adecuado
    level2 = np.concatenate((v_img[2], blanco2), axis=1)
    level3 = np.concatenate((v_img[3], blanco3), axis=1)
    level4 = np.concatenate((v_img[4], blanco4), axis=1)
    
    # Rellenamos filas y/o columnas
    # que no han quedado igualadas con los fondos
    if level1.shape[1] != level2.shape[1]:
        dif = level1.shape[1] - level2.shape[1]
        relleno = np.ones((level2.shape[0], dif))*255
        level2 = np.concatenate((level2, relleno), axis = 1)
    if level1.shape[1] != level3.shape[1]:
        dif = level1.shape[1] - level3.shape[1]
        relleno = np.ones((level3.shape[0], dif))*255
        level3 = np.concatenate((level3, relleno), axis = 1)
    if level1.shape[1] != level4.shape[1]:
        dif = level1.shape[1] - level4.shape[1]
        relleno = np.ones((level4.shape[0], dif))*255
        level4 = np.concatenate((level4, relleno), axis = 1)    
    
    # Concatenamos toda la parte derecha de la pirámide
    derecha = np.concatenate((level1, level2, level3, level4), axis=0)
    
    # Ajustamos la longitud de los ejes de la parte derecha
    if level0.shape[0] != derecha.shape[0]:
        dif = level0.shape[0] - derecha.shape[0]
        relleno = np.ones((dif, derecha.shape[1]))*255
        derecha = np.concatenate((derecha, relleno), axis = 0)
    
    # Unimos la parte derecha con el nivel 0
    final = np.concatenate((level0, derecha), axis=1)
    
    # Devolvemos la pirámide
    return final

def GaussianPyramid(im):    
    im2 = np.copy(im)
    v_imgs = []
    for i in range(5):
        im2 = Gaussiana2D(im2, 19, 3, cv2.BORDER_DEFAULT)
        v_imgs.append(im2)
        im2 = cv2.pyrDown(im2)
    p = pyramid(v_imgs)
    plt.figure(num=None, figsize=(15,15)) # Establecemos el tamaño de la imagen
    plt_imshow(p)
    plt.show()

im = leer_imagen('imagenes/Tablero1.jpg')
GaussianPyramid(im)