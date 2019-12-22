# Ejercicio 1

# Apartado A

> Pirámide Gaussiana

- Podemos usar ```pyrDown```.  
- Detección de punto harrys en cada uno de los puntos de la pirámide:  
  - Podemos usar ```cornerEigenValsandVecs(img, blocksize, ksize)```.
    - nos da $\lambda_1$, $\lambda_2$, y1, y1, x2, y2 (x e y, son autovectores).
    - Tenemos que controlar cuando $\lambda_x$ sea 0. Si esto ocurre, ponemos 0 directamente:
    - $fp=det(Mp)/traza(MP)=|\lambda1*\lambda2 / (\lambda1 + \lambda2)$
    - ¿de qué matriz calculamos los valores singulares?
      - Tenemos que buscar en las derivadas de la imagen. 
    - 7x7 es demasiado grande, tiene que ser común a todos los niveles de la pirámide. 
- Nos quedamos con los valores de fp mayores que un umbral (IMPORTANTE), empezamos sin umbral y según nos número que nos saque, o vamos aumentadno (fp > threshold).
- Ahora a los que nos hayan quedado depués de aplicar el umbral, hacemos la supresión de **no máximos** -> (winsize*winsize).
- Cómo modifican todos los parámetros, el número de puntos que se obtienen al final y explicar que esta haciendo cada parámetro. 
- Lo hacemos para todos los niveles de la pirámide. tenemos que ir guardando los keypoints que vayamos obteniendo (x,y,escala,orientación). KeyPoints es una clase de OpenCV, hay que usarla para luego poder trabajar con ellos, un keypoint por cada uno de los puntos que sobrevivan después de la supresión de no máximos. Cada fp es relativo al tamaño de la escala de donde lo hemos obtenido, por tanto, para conocer las coordenadas relativas de los keypoints en la imagen original, tenemos que hacer la transformación y calcularlos. 
  - ¿Cómo lo hacemos? Hemos eliminado las filas y columnas pares con pydown, por tanto, en la imagen real, un i,j en la siguiente escala, en la anterior será 2i, 2j. En la siguiente será 4i,4j, en la siguiente 8i,8j...
  - Aparte de guardar los píxeles, guardamos la escala y la orientación. La escala es fácil: ```cornerEigenValsandVecs``` el tamaño de blocksize que es el de la ventana, tenemos que multiplicarla por el nivel de la pirámide en el que esté. Para la orientación, tenemos que hacer: orientación=$\theta$ tal que $[cos \theta, sen \theta] = u / |u|$, u =(u1,u2). Alisamiento con sigma = 4.5. Tenemos dos imágenes por tanto, tenemos que calcular la imagen original con sus gradientes y le aplicamos el alisamiento con 4.5. Luego saco sus pirámides Gaussianas, una por cada gradiente, en cada nivel usaré el gradiente que me toque: Así calculamos la orientación. La orientación de cada keypoint es específica para cada uno y para cada escala. U1 y U2 son los puntos, después de haber alisado luego segun la derivada en X y segun la derivada en Y. Con el coseno y el seno, hallamos la tangente y podemos calcular el ángulo con la arcotangente. Cuidado seǵun la funcion de arcotangente que suamos, que nos de bien el cuadrante donde está el ángulo. Porque la función arcotangete es periódria entre -pi/2 y pi/2. Puede que solo obtengamos ángulos entre el segundo y el cuarto cuadrante, dependerá del signo del seno y el coseno. Por tanto, cuidado con los signos y con el cuadrante correcto.

Al final de todo tendremos una lista de keypoints que he ido obteniendo en todas las escalas. 

### Apuntes varios

- Supresión de no máximos en cada escala, el algoritmo de harrys en cada escala. El nivel de pirámide 1 es la imagen original.  
- Ir sacando funciones para modularizar.  
- Ir diciendo muy bien en la memoria qué vamos haciendo.  

# Apartado B

Modificar los parámetros para que sean los puntos representativos de la imagen. ¿por qué es representativo en la imagen?

# Apartado C

Mostramos los puntos representativos con ```drawKeyPoints()```. En la original, sacamos una imagen con los keypoints de cada uno de los niveles, así vemos cuales hay en cada escala. Luego mostramos todos juntos sobre la imagen original.

# Apartado D

Refinar las coordenadas x,y de los keypoints. Refinamos para coger adecuadamente el punto: podemos usar ```cornerSubPix()``` tiene parámetros que tenemos que modificar, explicar muy bien cómo funciona la función. Nos devolverá las coordenadas de las esquinas. Así tenemos los keypoins de antes y los de después. Vemos 3 puntos de ellos que no coincidan, cogemos tamaño 10x10 alrededor de esos puntos, hacemos un zoom de 5 y pintamos punto de antes y punto nuevo. Pintar en verde donde está bien (despues de refinar) y en rojo el que esta mal (antes de refinar). Usamos ```circle``` para la representación y rodear el píxel.  


# CUIDADO

Cuidado con pasar de radianes a grados o viceversal.


# 12 diciembre 2019

## Segundo apartado

Usar AKAZE. Criterios para hacer los matches:
    1. Fuerza bruta: cuál es el descriptor más cercano y luego hago cross check, es decir lo hago al revés. Si coinciden los matches, los dejo, si no, los tengoq ue eliminar. Me quedo solo con los más cercanos que lo sean tanto para una imagen como para la otra.   
    2. Para cada keypoint, cogemos los dos más cercanos en la primera. Sólo nos quedamos con aquel PRIMER match si y solo si está lo suficientemente separado con respecto al segundo. Hay que mirar en el paper de Lowe, el ratio de distancia, creo que es 0.8, pero podemos modificarlo según nos interese en nuestro ejercicio: $d(best) < 0.8*d(sig)$.

Luego cogemos 100 matches y los mostramos. Después tenemos que comparar ambos criterios de correspondencia. 


# EJERCICIO 2

Mostrar 100 matches ALEATORIOS.

BFMatcher(crosscheck=True) lo hace directamente.


### DMATCH
Para hallar las listas de puntos que se corresponden. Estas listas las tiene el objetivo DMATCH. query es la imagen/listakp? que le pasamos primero y train la que le pasamos después. Para cada match tendremos en cada lista un número que se corresponde al indice de la lista de keypoints. 

- trainIdx
- queryIdx
- distance

# EJERCICIO 3

Dos imagenes:

Ejercicio 2:
Usar umbral de Akaze si nos a ido bien, cogemos el mismo en este ejercicio si usamos las mismas imágenes.
1. KEYPOINTS.
2. MATCHES entre imágenes.
3. Comparación según a nuestros criterios y el mejor lo usams en el ejercicio 3. 

Ahora hay que usar:
- findHomography() para encontrar la homog de una imagen en la otra, direccion que usemos. No lo podemos pasar directamente keypoints de una y de otra (imágenes), hay que ordenador en funcion de los matches que tengamos. Akaze devuelve kp conforme los encuentra, igual que los matches. Nos vamos a los matches  miramos trainidx y queryid (me da índice de un kp y en trainidx me da indice de kp en base a la lista de la imagen trainidx. El match guarda el indice en la lista de query el indice del kp en la primera image, y guarda el indice del kp2 en la lista de trainkp. Tenemos que ordenar ambas listas en el orden de los matches: Podemos las posiciones de kp1 y kp2 en base a los matches). Tendremos listas de kp pero solo nos tenemos que quedar con el punto: pt. Tendremos dos listas de puntos correspondidas según matches. Esto se lo pasamos a finHomog junto con la imagen.  
Si tengo dos imagenes y un mosaico, tenemos que tomar una iamgen en negro donde voy a pintar el mosaico, entonces dentro pongo la primera imagen, es decir, la primera homografia. La primera homografía es una traslacion simple, luego colocamos la otra homografia donde queramos??????  
La primera homografia es una matriz 3x3 con una traslación. 
La segunda homografía: composción de dos homografias que es el producto de las dos matrices: las dos matrices son, la que lleva imagen 2 a imagen 1 y la otra homografia es la que lleva la imagen 1 a la traslación.

- WarpPerspective() para pasarlas en mosaico.


# EJERCICIO 4

¿Como minimizar errores al ir añadiendo homografías en el mosaico? Empezando por el centro y se deformarán la mitad a un lado y la mitad al otro lado. Es muy importante sacar las homografías de una imagen a otra (entre imágenes consecutivas). 