import random
import ayudaDirectorios
from calculoRangos import calcularRangos, rangosPrefijados, rangosImagenesIndividuales, rangosImagenesAleatorias
from diccionarioGlobal import calculaDiccionario
from normalizacionComparadores import obtieneMaxComparadores, sacarLista, aplicaNormalizacion


#1.Se obtienen los subdirectorios de el Directorio general "directorios"
directorios = ayudaDirectorios.obtenerDirectorios()

#1. Se recorre cada subdirectorio y se añade el path de cada imagen a la lista imagenes
imagenes=[]
for directorio in directorios:
    print("recorro el directorio "+directorio)
    for img_path in ayudaDirectorios.getAllFilesInDirectory("directorios/"+directorio):
        imagenes.append(img_path)

#3.Se eligen los diferentes métodos a emplear mediante la estructura flags
#flags
# 0 escalagrises
# 1 normalizado
# 2 clahe
# 3 hog
# 4 gabor
# 5 sift_sim
# 6 ssim
# 7 mse
# 8 gabor + sift sim
# 9 psnr
# 10 lsh
# 11 histograma color

#*****************************************ELEGIR FLAGS******************************************************************
flags=[1,1,1,1,1,1,1,1,1,1,1,1]
if (len(flags) !=11):
    raise AssertionError()
#***********************************************************************************************************************


#4. Se elige la forma de obtener los rangos:
# a)- Rangos prefijados
# b)- Rangos obtenidos para cada imagen mediante su comparativa con modificaciones automáticas para esa imagen
# c)- Rangos obtenidos para un número prefijado de imágenes del dataset  mediante su comparativa con modificaciones
#       automáticas para esas imágenes

#****************************************ELEGIR OPCIÓN RANGOS***********************************************************
# Elegir opcion
opcion="c"
#***********************************************************************************************************************
if(opcion=="a"): rangosPrefijados(flags)
elif(opcion=="b"): rangosImagenesIndividuales(flags)
elif(opcion=="c"):
    muestras=3
    rangosImagenesAleatorias(flags, muestras, directorios)
else: raise AssertionError()

#5. Tras calcular los rangos de similitud por comparador se procede a comparar cada imagen de un directorio con las demás
# imágenes dando como resultado un diccionario con las posibles copias
# tambien se genera un archivo txt con un informe de los posibles plagios
diccionarioGlobal=calculaDiccionario(imagenes, flags)
ayudaDirectorios.pretty(diccionarioGlobal)



