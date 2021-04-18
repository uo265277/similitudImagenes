import numpy as np
import ayudaDirectorios
from skimage import measure
import cv2
import os
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from imageio import imread
#*******************************************************************************
def sift_sim(path_a, path_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''

  # initialize the sift feature detector
  orb = cv2.ORB_create()
  # get the images
  print("estoy en sift sim con los path "+ path_a + "     "+ path_b)
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)
  print("he leido las imagenes")
  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)
  print( kp_a)
  print( kp_b)
  print("busco los keypoints")
  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)



#devuelve 1 de las 8 diferentes normas de matriz o vector
#¿aqui podria usar la biblioteca annoy para calcular el vecino mas cercano? o mas bien en otra funcion buscando entre todos los v. carac ya que aqui solo
#se comparan el v.caract de img a y de img b
#¿que norma es manhattan o la distancia euclidea? ¿cual es mejor?
from comparativas_sme_ssim import mse


def findDifference(f1, f2):
    return np.linalg.norm(f1-f2)
    #return manhattan_distance(f1, f2)ç

def distanciaCoseno(f1,f2):
  a1 = np.squeeze(np.asarray(f1))
  a2 = np.squeeze(np.asarray(f2))

  return ( round(np.dot(a1, a2) / (np.linalg.norm(f1) * np.linalg.norm(f2))  ,4) )

def manhattan_distance(a, b):
    return np.abs(a - b).sum()

def SSIM(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
    return s

#NO VA, ALGO DE DIMENSIONES
def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.
        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.
        Args:
            input1, input2: two input numpy arrays.
        Returns:
            Element-wise cosine distances of two inputs.
        '''
        return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
        #return np.dot(input1, input2.T) / \
         #       np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
          #              np.linalg.norm(input2.T, axis=0, keepdims=True))



#*******************************************************************************

# *******************************************************************************





def calculaDiccionarioDistancia(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    distancias: dict = {}
    for k in feature_vectors:
        print("Se procede a calcular las distancias de las demás imagenes con la imagen: " ,k)
        nombreDiccionario ="distacias " +k
        print(nombreDiccionario)
        nombreDiccionario: dict = {}
        distancias[k ]= nombreDiccionario
        for k2 in feature_vectors:


            if(k !=k2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                nombreDiccionario[k2 ] =diff

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return distancias


# *******************************************************************************


# igual que el anterior pero solo se comparan si estan en directorios diferentes
def calculaDiccionarioDistancia2(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    distancias: dict = {}
    for k in feature_vectors:
        dirK = ayudaDirectorios.directorioImagen(k)
        print("Se procede a calcular las distancias de las demás imagenes con la imagen: " ,k)
        nombreDiccionario ="distacias " +k
        print(nombreDiccionario)
        nombreDiccionario: dict = {}
        distancias[k ]= nombreDiccionario
        for k2 in feature_vectors:

            dirK2 = ayudaDirectorios.directorioImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                nombreDiccionario[k2 ] =diff

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return distancias


# *******************************************************************************


# igual que el anterior pero ordenando por diccionarios
def calculaDiccionarioDistancia3(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in feature_vectors:


        # CREO EL DICCIONARIO DE DIRECTORIO
        dirK = ayudaDirectorios.directorioImagen(k)

        # si estoy en el mismo directorio
        if cont==0:
            print("traza  if cont==0: ")
            cont =cont +1
            dirAux =dirK
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        print()
        print()
        print()
        print()
        print()
        print("*************DICCIONARIO DE DIRECTORIO  " +dirK +" ******************")
        # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

        # si el contador es 0 quiere decir que he cambiado de directorio
        if (dirAux!=dirK):
            print("traza if (dirAux!=dirK): ")
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}
            dirAux =dirK

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        # CREO EL SUBDICCIONARIO DE LA IMAGEN
        nomK = ayudaDirectorios.nombreImagen(k)
        diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
        print()
        print("--------------  " +diccionarioImagen +" -----------------------------")
        diccionarioImagen: dict = {}

        # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
        diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
        # print("Estado del diccionadio de directorios "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in feature_vectors:

            dirK2 = ayudaDirectorios.directorioImagen(k2)
            nomK2 = ayudaDirectorios.nombreImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                # diff=distanciaCoseno(feature_vectors[k],feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                diccionarioImagen[dirK2 +nomK2 ] =diff
                # print("hola2")
                # print(diccionarioImagen)
                # print("Estado del diccionadio de directorios "+ dirK+" : ")
                # print(diccionarioDirectorios)
                # print("fin")

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal


# *******************************************************************************


# igual que el anterior pero filtrand con el ratio

def calculaDiccionarioDistancia4(feature_vectors, ratio):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in feature_vectors:


        # CREO EL DICCIONARIO DE DIRECTORIO
        dirK = ayudaDirectorios.directorioImagen(k)

        # si estoy en el mismo directorio
        if cont==0:
            print("traza  if cont==0: ")
            cont =cont +1
            dirAux =dirK
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        print()
        print()
        print()
        print()
        print()
        print("*************DICCIONARIO DE DIRECTORIO  " +dirK +" ******************")
        # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

        # si el contador es 0 quiere decir que he cambiado de directorio
        if (dirAux!=dirK):
            print("traza if (dirAux!=dirK): ")
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}
            dirAux =dirK

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        # CREO EL SUBDICCIONARIO DE LA IMAGEN
        nomK = ayudaDirectorios.nombreImagen(k)
        diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
        print()
        print("--------------  " +diccionarioImagen +" -----------------------------")
        diccionarioImagen: dict = {}

        # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
        diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
        # print("Estado del diccionadio de directorios "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in feature_vectors:

            dirK2 = ayudaDirectorios.directorioImagen(k2)
            nomK2 = ayudaDirectorios.nombreImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)

                if(diff<=ratio):
                    diccionarioImagen[dirK2 +nomK2 ] =diff
                # print("hola2")
                # print(diccionarioImagen)
                # print("Estado del diccionadio de directorios "+ dirK+" : ")
                # print(diccionarioDirectorios)
                # print("fin")

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal















#
# def calculaDiccionarioDistancia5(imagenes, ratio):
#     # entrada -> clave valor imagen y su vector
#     # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea
#
#     # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
#     # CREO EL DICCIONARIO GLOBAL
#     diccionarioGlobal: dict = {}
#     cont =0
#     dirAux =""
#
#     for k in imagenes:
#
#
#         # CREO EL DICCIONARIO DE DIRECTORIO
#         dirK = ayudaDirectorios.directorioImagen(k)
#
#         # si estoy en el mismo directorio
#         if cont==0:
#             print("traza  if cont==0: ")
#             cont =cont +1
#             dirAux =dirK
#             diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
#             print(diccionarioDirectorios)
#             diccionarioDirectorios: dict = {}
#
#             # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
#             diccionarioGlobal[dirK ]= diccionarioDirectorios
#
#         print()
#         print()
#         print()
#         print()
#         print()
#         print("*************DICCIONARIO DE DIRECTORIO  " +dirK +" ******************")
#         # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)
#
#         # si el contador es 0 quiere decir que he cambiado de directorio
#         if (dirAux!=dirK):
#             print("traza if (dirAux!=dirK): ")
#             diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
#             print(diccionarioDirectorios)
#             diccionarioDirectorios: dict = {}
#             dirAux =dirK
#
#             # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
#             diccionarioGlobal[dirK ]= diccionarioDirectorios
#
#         # CREO EL SUBDICCIONARIO DE LA IMAGEN
#         nomK = ayudaDirectorios.nombreImagen(k)
#         diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
#         print()
#         print("--------------  " +diccionarioImagen +" -----------------------------")
#         diccionarioImagen: dict = {}
#
#         # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
#         diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
#         # print("Estado del diccionadio de directorios "+ dirK+" : ")
#         # print(diccionarioDirectorios)
#         # print("fin")
#         for k2 in imagenes:
#
#             dirK2 = ayudaDirectorios.directorioImagen(k2)
#             nomK2 = ayudaDirectorios.nombreImagen(k2)
#             if(dirK !=dirK2):
#                 imagen1=cv2.imread(k)
#                 imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
#                 imagen1 = cv2.resize(imagen1, (224, 224))
#
#                 imagen2=cv2.imread(k2)
#                 imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
#                 imagen2 = cv2.resize(imagen2, (224, 224))
#                 diff =SSIM(imagen1,imagen2)
#                 print("El SSIM de ", k, " con " ,k2, "es", diff)
#
#                 if(diff>=ratio):
#                     diccionarioImagen[dirK2 +nomK2 ] =diff
#                 # print("hola2")
#                 # print(diccionarioImagen)
#                 # print("Estado del diccionadio de directorios "+ dirK+" : ")
#                 # print(diccionarioDirectorios)
#                 # print("fin")
#
#         # posible filtro aqui mas tarde
#         #  print("diff es")
#         #  print(diff)
#         #  distancias[k]=diff
#     return diccionarioGlobal
#
#


def calculaDiccionarioDistancia6(imagenes, ratio):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in imagenes:


        # CREO EL DICCIONARIO DE DIRECTORIO
        dirK = ayudaDirectorios.directorioImagen(k)

        # si estoy en el mismo directorio
        if cont==0:
            print("traza  if cont==0: ")
            cont =cont +1
            dirAux =dirK
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        print()
        print()
        print()
        print()
        print()
        print("*************DICCIONARIO DE DIRECTORIO  " +dirK +" ******************")
        # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

        # si el contador es 0 quiere decir que he cambiado de directorio
        if (dirAux!=dirK):
            print("traza if (dirAux!=dirK): ")
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}
            dirAux =dirK

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        # CREO EL SUBDICCIONARIO DE LA IMAGEN
        nomK = ayudaDirectorios.nombreImagen(k)
        diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
        print()
        print("--------------  " +diccionarioImagen +" -----------------------------")
        diccionarioImagen: dict = {}

        # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
        diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
        # print("Estado del diccionadio de directorios "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in imagenes:
            dirK2 = ayudaDirectorios.directorioImagen(k2)
            nomK2 = ayudaDirectorios.nombreImagen(k2)
            if(dirK !=dirK2):
                #saco los path absolutos porque isno da error con los relativos
                diff =sift_sim(k,k2)
                print("El sift sim de ", k, " con " ,k2, "es", diff)

                if(diff>=ratio):
                    diccionarioImagen[dirK2 +nomK2 ] =diff
                # print("hola2")
                # print(diccionarioImagen)
                # print("Estado del diccionadio de directorios "+ dirK+" : ")
                # print(diccionarioDirectorios)
                # print("fin")

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal
