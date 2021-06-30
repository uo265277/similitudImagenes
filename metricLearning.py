"""
Title: Metric learning for image similarity search
Author: [Mat Kelcey](https://twitter.com/mat_kelcey)
Date created: 2020/06/05
Last modified: 2020/06/09
Description: Example of using similarity metric learning on CIFAR-10 images.
"""


"""
## Overview

Metric learning aims to train models that can embed inputs into a high-dimensional space
such that "similar" inputs, as defined by the training scheme, are located close to each
other. These models once trained can produce embeddings for downstream systems where such
similarity is useful; examples include as a ranking signal for search or as a form of
pretrained embedding model for another supervised problem.

For a more detailed overview of metric learning see:

* [What is metric learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
* ["Using crossentropy for metric learning" tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)
"""

"""
## Setup
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

#apaño para un error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#fin apaño

#para lanzarlo desatendido ponerlo a False, de lo contrario se para con los plot
#alternativamente, guardar a imagen, pero entonces se pierde poder hacer zoom en la matriz
#de confusión
LOSS=False
CMATRIX=False


from imutils import paths
import cv2
import os
import random
import numpy as np
#Leer todas las imágenes de un directorio
carpeta="directorios/TODOS_TRABAJOS"

# grab the image paths and randomly shuffle them
print("[INFO] Cargando imagenes...")
#las imágenes de train y test están en este directorio
imagePaths = sorted(list(paths.list_images(carpeta)))

#algunos hiperparámetros,
#TODO en realidad los rangos de variación de recorte, color, n batches epochs, etc también serían hiperparámetros
# los que mejor resultado me han dado para la
#red original son estos IMAGE_DIMS = (128,128, 3) tamEmbedding=128 pctTrain=0.7 nVariaciones=1000
#tamaño de las imágenes del dataset, tiene que coincidir con el de la red
#cuando se haga inferencia hay que escalar las imaǵenes a este tamaño
IMAGE_DIMS = (128,128, 3)
#tamaño de la representación latente
tamEmbedding=128
pctTrain=0.7
#cuantas variaciones aleatorias se hacen de cada imagen original para generar el dataset de ejemplos
nVariaciones=2000

# original 1000 20
num_batchs=500
epochs=20

#porcentaje minimo crop


#leer las imágenes originales en un diccionario nombre:imagen
imagenes={}
#carga imágenes en un diccionario, la clave es el nombre sin extensión
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    label = ".".join(imagePath.split(os.path.sep)[-1].split('.')[:-1])
    imagenes[label]=image

#generar variaciones random de cada imagen para componer el dataset
#TODO quizás sea mejor a la hora de hacer experimentos y hacer visualizaciones
# guardarlo todo a disco
#brillo, contraste, color, recorte, traslacion, censura, giro, suavizado, etc
#he implementado recorte, color, brillo, suaviza, censura
#se queda entre pctx/y ->1
def recorte(img,pctx,pcty):
    #tamaño recortado
    rx=int(img.shape[0]*random.uniform(pctx,1))
    ry=int(img.shape[1]*random.uniform(pcty,1))
    #esquina sup izq recorte
    offsetx=random.randint(0,img.shape[0]-rx)
    offsety=random.randint(0,img.shape[1]-ry)
    return img[offsetx:offsetx+rx,offsety:offsety+ry,:]
#pct maximo de variacion
#internamente se modifica aleatoriamente cada banda
#dentro de ese +-rango
def color(img,pct):
    #extraer bandas
    (blue,green,red)=cv2.split(img)
    #incrementos aleatorios
    deltar=random.randint(int(-255*pct),int(255*pct))
    deltag=random.randint(int(-255*pct),int(255*pct))
    deltab=random.randint(int(-255*pct),int(255*pct))
    #modificar bandas, en esencia máscaras para lo que queda a cero, lo que se modifica y lo que queda a 255
    redEnhanced=np.logical_and((red+deltar>=0),(red+deltar)<=255).astype(np.uint8)*(red+deltar)+\
                (red+deltar<0).astype(np.uint8)*0+\
                (red+deltar>255).astype(np.uint8)*255
    greenEnhanced=np.logical_and((green+deltag>=0),(green+deltag)<=255).astype(np.uint8)*(green+deltag)+\
                  (green+deltag<0).astype(np.uint8)*0+\
                  (green+deltag>255).astype(np.uint8)*255
    blueEnhanced=np.logical_and((blue+deltab>=0),(blue+deltab)<=255).astype(np.uint8)*(blue+deltab)+\
                 (blue+deltab<0).astype(np.uint8)*0+\
                 (blue+deltab>255).astype(np.uint8)*255
    return  cv2.merge([blueEnhanced.astype(np.uint8),greenEnhanced.astype(np.uint8),redEnhanced.astype(np.uint8)])

def brillo(img,pct):
    #extraer bandas
    (blue,green,red)=cv2.split(img)
    #incremento aleatorio
    delta=random.randint(int(-255*pct),int(255*pct))
    #modificar bandas, en esencia máscaras para lo que queda a cero, lo que se modifica y lo que queda a 255
    redEnhanced=np.logical_and((red+delta>=0),(red+delta)<=255).astype(np.uint8)*(red+delta)+\
                (red+delta<0).astype(np.uint8)*0+\
                (red+delta>255).astype(np.uint8)*255
    greenEnhanced=np.logical_and((green+delta>=0),(green+delta)<=255).astype(np.uint8)*(green+delta)+\
                  (green+delta<0).astype(np.uint8)*0+\
                  (green+delta>255).astype(np.uint8)*255
    blueEnhanced=np.logical_and((blue+delta>=0),(blue+delta)<=255).astype(np.uint8)*(blue+delta)+\
                 (blue+delta<0).astype(np.uint8)*0+\
                 (blue+delta>255).astype(np.uint8)*255
    return  cv2.merge([blueEnhanced.astype(np.uint8),greenEnhanced.astype(np.uint8),redEnhanced.astype(np.uint8)])
#suaviza, con un kernel todo unos
#t define el tamaño máximo, 1->3, 2->5, 3->7
def suaviza(img,t):
    #para que pueda salir aleatoriamente que no se suaviza
    randomInt=random.randint(0,t)
    if randomInt>=1:
        tKernel=random.randint(1,t)*2+1;
        return cv2.blur(img,(tKernel,tKernel))
    else:
        return img[:,:,:]
#superpone un rectángulo negro en un sitio aleatorio
def censura(img,alto,ancho):
    i=random.randint(0,img.shape[0]-alto)
    j=random.randint(0,img.shape[1]-ancho)
    #se modifica una copia, no el original
    copia=img[:,:,:]
    copia[i:i+alto,j:j+ancho,:]=0
    return copia
#generación del dataset
print("[INFO] Generando dataset...")
y_test=[]
x_test=[]
y_train=[]
x_train=[]
#indices de los ejemplos de train
index_train=random.sample(range(len(imagenes)*nVariaciones),int(len(imagenes)*nVariaciones*0.8))
#contador para saber en donde va cada ejemplo test o train
k=0
porcentajeMinimoCrop=0.7
#cada imagen original define su clase, recorro en paralelo
#la lista de nombres y un rango de modo que la primera imagen es la clase 0, la segunda l la 1, etc
for nombre,clase in zip(imagenes.keys(),range(len(imagenes))):
    #de cada original hago variaciones que pertenecen a la clase de la original
    for i in range(nVariaciones):
        #print(i,nombre)
        #modificación
        #TODO quizás para el estudio se puedan hacer variaciones de cada
        #clase y mezcladas
        imagenModificada=recorte(imagenes[nombre], porcentajeMinimoCrop, porcentajeMinimoCrop)
        #al introducir más variaciones se complica el reconocimiento de
        #copias si estas modificaciones no están en las que se buscan
        #imagenModificada=color(imagenModificada,.2)
        #imagenModificada=brillo(imagenModificada,.2)
        #imagenModificada=suaviza(imagenModificada,2)
        if k in index_train:
            #se añade con el tamaño de la red
            x_train.append(cv2.resize(imagenModificada, (IMAGE_DIMS[1], IMAGE_DIMS[0])))
            y_train.append(clase)
        else:
            x_test.append(cv2.resize(imagenModificada, (IMAGE_DIMS[1], IMAGE_DIMS[0])))
            y_test.append(clase)
        k+=1


x_train = np.array(x_train).astype("float32") / 255.0
y_train = np.squeeze(y_train)
x_test = np.array(x_test).astype("float32") / 255.0
y_test = np.squeeze(y_test)


"""
To get a sense of the dataset we can visualise a grid of 25 random examples.


"""

#Las imágenes tienen que ser cuadradas, un aserto por si pasa algo raro
assert (IMAGE_DIMS[0]==IMAGE_DIMS[1])

#esto es el tamaño de la entrada de la red == al de las imágenes de ejemplo
height_width = IMAGE_DIMS[0]


def show_collage(examples):
    box_size = height_width + 2
    num_rows, num_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size, num_rows * box_size),
        color=(250, 250, 250),
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    # Double size for visualisation.
    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage




"""
Metric learning provides training data not as explicit `(X, y)` pairs but instead uses
multiple instances that are related in the way we want to express similarity. In our
example we will use instances of the same class to represent similarity; a single
training instance will not be one image, but a pair of images of the same class. When
referring to the images in this pair we'll use the common metric learning names of the
`anchor` (a randomly chosen image) and the `positive` (another randomly chosen image of
the same class).

To facilitate this we need to build a form of lookup that maps from classes to the
instances of that class. When generating data for training we will sample from this
lookup.
"""

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

"""
For this example we are using the simplest approach to training; a batch will consist of
`(anchor, positive)` pairs spread across the classes. The goal of learning will be to
move the anchor and positive pairs closer together and further away from other instances
in the batch. In this case the batch size will be dictated by the number of classes; for
CIFAR-10 this is 10.
"""

#el número de clases es el número de imágenes en el documento fuente
num_classes = len(imagenes)


class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, height_width, height_width, 3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train[anchor_idx]
            x[1, class_idx] = x_train[positive_idx]
        return x


"""
We can visualise a batch in another collage. The top row shows randomly chosen anchors
from the 10 classes, the bottom row shows the corresponding 10 positives.
"""

examples = next(iter(AnchorPositivePairs(num_batchs=1)))

#plt.imshow(show_collage(examples))
#plt.savefig("collage.jpg")
"""
## Embedding model

We define a custom model with a `train_step` that first embeds both anchors and positives
and then uses their pairwise dot products as logits for a softmax.
"""


class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


"""
Next we describe the architecture that maps from an image to an embedding. This model
simply consists of a sequence of 2d convolutions followed by global pooling with a final
linear projection to an embedding space. As is common in metric learning we normalise the
embeddings so that we can use simple dot products to measure similarity. For simplicity
this model is intentionally small.
"""

# red original
# inputs = layers.Input(shape=(height_width, height_width, 3))
# x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs)
# x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x)
# x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x)
# x = layers.GlobalAveragePooling2D()(x)
# embeddings = layers.Dense(units=tamEmbedding, activation=None)(x)
# embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
#
# model = EmbeddingModel(inputs, embeddings)

#usando transfer learning y xCeption no nos da la memoria
#no nos da la memoria
if False:
    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(height_width, height_width, 3),
        include_top=False)
    base_model.trainable = False

    inputs = keras.Input(shape=(height_width, height_width, 3))
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(units=tamEmbedding, activation=None)(x)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    model = EmbeddingModel(inputs, embeddings)
#https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
if False:
    from keras.applications import MobileNet
    from keras.layers import Dense,GlobalAveragePooling2D,Conv2D

    base_model=MobileNet(weights='imagenet',
                        input_shape=(height_width, height_width, 3),
                        include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
    #base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable=False

    x=base_model.output
    x=GlobalAveragePooling2D()(x)

    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3

    embeddings = Dense(units=tamEmbedding, activation=None)(x)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    model = EmbeddingModel(base_model.inputs, embeddings)
    #o base_model.trainable = False más arriba o esto
    #for layer in model.layers:
    #    layer.trainable=False


if True:
    inputs = layers.Input(shape=(height_width, height_width, 3))
    x = layers.Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(units=tamEmbedding, activation=None)(x)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    model = EmbeddingModel(inputs, embeddings)

"""
Finally we run the training. On a Google Colab GPU instance this takes about a minute.
"""

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(AnchorPositivePairs(num_batchs=num_batchs), epochs=epochs)

if LOSS:
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

"""
## Testing

We can review the quality of this model by applying it to the test set and considering
near neighbours in the embedding space.

First we embed the test set and calculate all near neighbours. Recall that since the
embeddings are unit length we can calculate cosine similarity via dot products.
"""

near_neighbours_per_example = 10

embeddings = model.predict(x_test)
gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

"""
As a visual check of these embeddings we can build a collage of the near neighbours for 5
random examples. The first column of the image below is a randomly selected image, the
following 10 columns show the nearest neighbours in order of similarity.
"""

num_collage_examples = 5

examples = np.empty(
    (
        num_collage_examples,
        near_neighbours_per_example + 1,
        height_width,
        height_width,
        3,
    ),
    dtype=np.float32,
)
for row_idx in range(num_collage_examples):
    examples[row_idx, 0] = x_test[row_idx]
    anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]

#plt.imshow(show_collage(examples))
#plt.savefig("collage2.jpg")

"""
We can also get a quantified view of the performance by considering the correctness of
near neighbours in terms of a confusion matrix.

Let us sample 10 examples from each of the 10 classes and consider their near neighbours
as a form of prediction; that is, does the example and its near neighbours share the same
class?

We observe that each animal class does generally well, and is confused the most with the
other animal classes. The vehicle classes follow the same pattern.
"""

confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider 10 examples.
    example_idxs = class_idx_to_test_idxs[class_idx][:10]
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_test_idx][:-1]:
            nn_class_idx = y_test[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1

# Display a confusion matrix.
#las etiquetas son el nombre de la imagen
labels = imagenes.keys()
if CMATRIX:
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
    plt.show()



embeddingsOriginales = model.predict(np.array([imagenes[imagen] for imagen in imagenes.keys()]))
diccionarioEmbeddingsOriginales={clave:valor for clave,valor in zip(imagenes.keys(),embeddingsOriginales)}

def modulo(v):
    return (sum([e*e for e in v]))**.5

def compruebaEmbeddings(embeddings):
    for e in embeddings:
        if abs(modulo(e)-1.0)>.001:
            return  False
    return True

print("Embeddings orig ok?",compruebaEmbeddings(embeddingsOriginales))
#leer las imágenes de otro directorio/documento, calcular la representación, calcular la diferencia
#con la representación de cada uno de los originales

carpeta='/home/claudia/PycharmProjects/similitudImagenes/directorios/TRABAJO_A'

# grab the image paths and randomly shuffle them
print("[INFO] Cargando imagenes...")
#las imágenes de train y test están en este directorio
imagePaths = sorted(list(paths.list_images(carpeta)))
otrasImagenes={}
print("Maximo de p escalar")
#carga imágenes en un diccionario, la clave es el nombre sin extensión
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    label = ".".join(imagePath.split(os.path.sep)[-1].split('.')[:-1])
    otrasImagenes[label]=image


def distanciaEuclidea(a,b):
    return sum([(a[i]-b[i])**2 for i in range(len(a))])**.5
def pEscalar(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])
embeddingsOtras = model.predict(np.array([otrasImagenes[imagen] for imagen in otrasImagenes.keys()]))
diccionarioEmbeddingsOtras={clave:valor for clave,valor in zip(otrasImagenes.keys(),embeddingsOtras)}
print("Embeddings otras ok?",compruebaEmbeddings(embeddingsOtras))
#Para cada imagen original, imagen mas parecide de las otras
for imagen in diccionarioEmbeddingsOriginales.keys():
    #minimo=float('Inf')
    maximo=-1
    for otraImagen in diccionarioEmbeddingsOtras.keys():
        #usar la d euclidea en lugar de la que se usó para entrenar tiene un sentido regular
        #d=distanciaEuclidea(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        #En teoría la distancia del coseno == pEscalar en este contexto.
        #-1 opuesto 0 distintas 1 iguales, entonces 1-pEscalar es la distancia o lo contrario del parecido
        d=pEscalar(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        if d>maximo:
            maximo=d
            masParecida=otraImagen
    #no tiene sentido pero parece que cuanto más cerca de 1 más parecido
    print(imagen,masParecida,d)
#gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)



carpeta='/home/claudia/PycharmProjects/similitudImagenes/directorios/TRABAJO_A'

# grab the image paths and randomly shuffle them
print("[INFO] Cargando imagenes...")
#las imágenes de train y test están en este directorio
imagePaths = sorted(list(paths.list_images(carpeta)))
otrasImagenes={}
#carga imágenes en un diccionario, la clave es el nombre sin extensión
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    label = ".".join(imagePath.split(os.path.sep)[-1].split('.')[:-1])
    otrasImagenes[label]=image

embeddingsOtras = model.predict(np.array([otrasImagenes[imagen] for imagen in otrasImagenes.keys()]))
diccionarioEmbeddingsOtras={clave:valor for clave,valor in zip(otrasImagenes.keys(),embeddingsOtras)}
#Para cada imagen original, imagen mas parecide de las otras
masParecidos=[]
for imagen in diccionarioEmbeddingsOriginales.keys():
    #minimo=float('Inf')
    maximo=-1
    for otraImagen in diccionarioEmbeddingsOtras.keys():
        #usar la d euclidea en lugar de la que se usó para entrenar tiene un sentido regular
        #d=distanciaEuclidea(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        #En teoría la distancia del coseno == pEscalar en este contexto.
        #-1 opuesto 0 distintas 1 iguales, entonces 1-pEscalar es la distancia o lo contrario del parecido
        d=pEscalar(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        if d>maximo:
            maximo=d
            masParecida=otraImagen
    #no tiene sentido pero parece que cuanto más cerca de 1 más parecido
    print(imagen,masParecida,d)




#gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
#for imagen in list(imagenes.keys())[:10]:
#    plt.plot([1-pEscalar(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen]) for otraImagen in diccionarioEmbeddingsOtras.keys()])
#plt.show()

#prueba más fácil: generar variaciones random y decir a que original se parece más

#demo del proyecto, seleccionar unas cuantas imágenes originales y editarlas para hacerlas pasar por propias
#mirar si las detecta entre otras no plagiadas.


carpeta='/home/claudia/PycharmProjects/similitudImagenes/directorios/TRABAJO_A'

# grab the image paths and randomly shuffle them
print("[INFO] Cargando imagenes...")
#las imágenes de train y test están en este directorio
imagePaths = sorted(list(paths.list_images(carpeta)))
otrasImagenes={}
#carga imágenes en un diccionario, la clave es el nombre sin extensión
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    label = ".".join(imagePath.split(os.path.sep)[-1].split('.')[:-1])
    otrasImagenes[label]=image

embeddingsOtras = model.predict(np.array([otrasImagenes[imagen] for imagen in otrasImagenes.keys()]))
diccionarioEmbeddingsOtras={clave:valor for clave,valor in zip(otrasImagenes.keys(),embeddingsOtras)}
#Para cada imagen original, imagen mas parecide de las otras
masParecidos=[]
for imagen in diccionarioEmbeddingsOriginales.keys():
    #minimo=float('Inf')
    maximo=-1
    for otraImagen in diccionarioEmbeddingsOtras.keys():
        #usar la d euclidea en lugar de la que se usó para entrenar tiene un sentido regular
        #d=distanciaEuclidea(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        #En teoría la distancia del coseno == pEscalar en este contexto.
        #-1 opuesto 0 distintas 1 iguales, entonces 1-pEscalar es la distancia o lo contrario del parecido
        d=pEscalar(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        if d>maximo:
            maximo=d
            masParecida=otraImagen
    #no tiene sentido pero parece que cuanto más cerca de 1 más parecido
    print(imagen,masParecida,d)




carpeta='/home/claudia/PycharmProjects/similitudImagenes/directorios/TRABAJO_A'

# grab the image paths and randomly shuffle them
print("[INFO] Cargando imagenes...")
#las imágenes de train y test están en este directorio
imagePaths = sorted(list(paths.list_images(carpeta)))
otrasImagenes={}
print("Minimo d euclidea")
#carga imágenes en un diccionario, la clave es el nombre sin extensión
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    label = ".".join(imagePath.split(os.path.sep)[-1].split('.')[:-1])
    otrasImagenes[label]=image

embeddingsOtras = model.predict(np.array([otrasImagenes[imagen] for imagen in otrasImagenes.keys()]))
diccionarioEmbeddingsOtras={clave:valor for clave,valor in zip(otrasImagenes.keys(),embeddingsOtras)}
#Para cada imagen original, imagen mas parecide de las otras
masParecidos=[]
for imagen in diccionarioEmbeddingsOriginales.keys():
    minimo=float('Inf')
    #maximo=-1
    for otraImagen in diccionarioEmbeddingsOtras.keys():
        #usar la d euclidea en lugar de la que se usó para entrenar tiene un sentido regular
        d=distanciaEuclidea(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        #En teoría la distancia del coseno == pEscalar en este contexto.
        #-1 opuesto 0 distintas 1 iguales, entonces 1-pEscalar es la distancia o lo contrario del parecido
        #d=pEscalar(diccionarioEmbeddingsOriginales[imagen],diccionarioEmbeddingsOtras[otraImagen])
        if d<minimo:
            minimo=d
            masParecida=otraImagen
    #no tiene sentido pero parece que cuanto más cerca de 1 más parecido
    print(imagen,masParecida,d)
