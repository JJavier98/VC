# -*- coding: utf-8 -*-

# Bibliotecas
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils
import numpy as np
from keras.models import Sequential # Modulos
from keras.layers import Dense, Dropout, Flatten # Capas
from keras.layers import Conv2D, MaxPooling2D	 # Capas
from keras.optimizers import SGD # Optimizadores
from keras.datasets import cifar100 # Conjunto de datos
from keras.preprocessing.image import ImageDataGenerator

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################
"""
A esta función sólo se le llama una vez.Devuelve 4
vectores conteniendo , por este orden , las im á genes
de entrenamiento , las clases de las imagenes de
entrenamiento , las im á genes del conjunto de test y
las clases del conjunto de test .
"""
def cargarImagenes ():
	"""
	Cargamos Cifar100.Cada imagen tiene tamaño
	(32 , 32 , 3). Nos vamos a quedar con las
	imágenes de 25 de las clases.
	"""
	( x_train , y_train ), ( x_test , y_test ) = cifar100.load_data( label_mode = 'fine')
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train /= 255
	x_test /= 255

	train_idx = np.isin ( y_train , np.arange (25))
	train_idx = np.reshape ( train_idx , -1)
	x_train = x_train [ train_idx ]
	y_train = y_train [ train_idx ]

	test_idx = np.isin ( y_test , np.arange (25))
	test_idx = np.reshape ( test_idx , -1)
	x_test = x_test [ test_idx ]
	y_test = y_test [ test_idx ]

	"""
	Transformamos los vectores de clases en matrices .
	Cada componente se convierte en un vector de ceros
	con un uno en la componente correspondiente a la
	clase a la que pertenece la imagen.Este paso es
	necesario para la clasificación multiclase en keras .
	"""

	y_train = np_utils.to_categorical ( y_train , 25)
	y_test = np_utils.to_categorical ( y_test , 25)

	return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################
"""
Esta función devuelve el accuracy de un modelo , defi -
nido como el porcentaje de etiquetas bien predichas
frente al total de etiquetas.Como par á metros es
necesario pasarle el vector de etiquetas verdaderas
y el vector de etiquetas predichas , en el formato de
keras ( matrices donde cada etiqueta ocupa una fila ,
con un 1 en la posici ó n de la clase a la que pertenece
0 en las dem á s ).
"""
def calcularAccuracy ( labels , preds ):
	labels = np.argmax ( labels , axis = 1)
	preds = np.argmax ( preds , axis = 1)
	accuracy = sum ( labels == preds )/ len ( labels )

	return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################
"""
Esta función pinta dos gr á ficas , una con la evoluci ó n
de la función de p é rdida en el conjunto de train y
en el de validaci ón , y otra con la evoluci ó n del
accuracy en el conjunto de train y el de validaci ó n .
Es necesario pasarle como par á metro el historial del
entrenamiento del modelo ( lo que devuelven las
funciones fit () y fit_generator ()).
"""
def mostrarEvolucion ( hist ):
	loss = hist.history ['loss']
	val_loss = hist.history ['val_loss']
	plt.plot ( loss )
	plt.plot ( val_loss )
	plt.legend (['Training loss','Validation loss'])
	plt.show () 

	acc = hist.history ['acc']
	val_acc = hist.history ['val_acc']
	plt.plot ( acc )
	plt.plot ( val_acc )
	plt.legend (['Training acc','Validation acc'])
	plt.show ()


#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
########################### EJERCICIO 1 #################################
#########################################################################

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5),
				 activation='relu',
				 input_shape=(32, 32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(14,14,6)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))


#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

optimizador = keras.optimizers.Adam()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizador,
              metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 32
epochs = 10

# HE CAMBIADO fit POR fit_generator
historia = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#input('Intro. Mostrar gráfica sobre la precisión y pérdida del modelo')
mostrarEvolucion(historia)

#########################################################################
########################## MEJORA DEL MODELO ############################
############################# EJERCICIO 2 ###############################
#########################################################################

#input('Intro. Continuar con ejercicio 2')

# Restaurar pesos
model.set_weights(weights)

# Definimos un generador de datos que divida el conjunto de entrenamiento en un 90%
# de entrenamiento y un 10% de validación
datagen = ImageDataGenerator(featurewise_center=True, validation_split = 0.1,
                             featurewise_std_normalization = True)
# Definimos otro generador de datos pero que solo contenga la normalizacióón de los datos.
datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization = True)

# Ajustamos a los datos de entrenamiento
datagen.fit(x_train)
datagen_test.fit(x_train)
# Definimos tamaño de batch y num de épocas
batch_size = 32
epochs = 10
model.fit_generator(datagen.flow(x_train ,y_train, batch_size = batch_size, subset = 'training'),
                    validation_data = datagen.flow(x_train, y_train ,batch_size = 32,subset ='validation'),
                    epochs = epochs,
                    steps_per_epoch = len(x_train)*0.9/batch_size,
                    validation_steps = len(x_train )*0.1/batch_size)


#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#input('Intro. Mostrar gráfica sobre la precisión y pérdida del modelo')

# CONTAR EN MEMORIA QUE USO EVALUATE EN VEZ DE PREDICT Y CALCULAR ACCURACY
score = model.evaluate(datagen_test.flow(x_test, y_test), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
mostrarEvolucion(historia)

























