from textwrap import indent

import ayudaDirectorios
from diccionarioGlobal import calculaDiccionario
from normalizacionComparadores import obtieneMaxComparadores, sacarLista, aplicaNormalizacion

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


#sacar


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


