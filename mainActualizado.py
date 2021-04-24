import os
import random
from textwrap import indent

import ayudaDirectorios
from calculoRangos import calcularRangos
from diccionarioGlobal import calculaDiccionario
from normalizacionComparadores import obtieneMaxComparadores, sacarLista, aplicaNormalizacion
#en esta parte se sacan con
#Se obtienen los subdirectorios de el Directorio general "directorios"
directorios = ayudaDirectorios.obtenerDirectorios()

#se recorre cada subdirectorio y se añade -> PATH_IMAGEN
imagenes=[]
for directorio in directorios:
    print("recorro el directorio "+directorio)
    for img_path in ayudaDirectorios.getAllFilesInDirectory("directorios/"+directorio):
        imagenes.append(img_path)
#flags
# 0 escalagrises
# 1 normalizado
# 2 clahe
# 3 hog
# 4 gabor
# 5 sift_sim
# 6 ssim
# 7 mse
# 8 gabor_sift_sim
flags=[1,1,1,0,1,1,0,1,0]


#sacar rangos con una imagen aletoria de un directorio, quiza la primera
dirAleatorio=random.choices(directorios)
dirAleatorio=dirAleatorio.pop()
pathAleatorio=random.choices(ayudaDirectorios.getAllFilesInDirectory("directorios/"+dirAleatorio))
pathAleatorio=pathAleatorio.pop()
print(pathAleatorio)
print("traza")
#siendo rangos una lista con los maximos de cada comparador
rangos= calcularRangos(pathAleatorio, flags, 0.05)

print("los rangos elegidos son:")
print(rangos)

diccionarioGlobal=calculaDiccionario(imagenes, flags)


ayudaDirectorios.pretty(diccionarioGlobal)

##############NORMALIZACION##############################################################################
#a este diccionario global hay que sacarle todas las listas de subdiccionarios y fusionarlos en una lista
#para sacar el maximo

lista=[]
#saco lso maximos de cada comparador
lista=sacarLista(diccionarioGlobal, lista)
maximos=obtieneMaxComparadores(lista, flags)
print(maximos)
#aplico el normalizado con esos maximos
diccionarioGlobalNormalizado=aplicaNormalizacion(diccionarioGlobal, maximos)
ayudaDirectorios.pretty(diccionarioGlobalNormalizado)


