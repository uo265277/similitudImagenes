import os
import random
import ayudaDirectorios
from calculoRangos import calcularRangos, rangosPrefijados, rangosImagenesIndividuales, rangosImagenesAleatorias
from diccionarioGlobal import calculaDiccionario
from normalizacionComparadores import obtieneMaxComparadores, sacarLista, aplicaNormalizacion
from subprocess import call
from pasaraJPG import pasarAJPG

#0. Se extraen las imágenes de los PDF del directorio pdf
#en caso de dar errores de permiso -> chmod 755 ejecutable
trabajos = ayudaDirectorios.obtenerTrabajos()
for pdf in trabajos:
    nombreDir="directorios/"+pdf+"/"
    nombretrabajo="pdf/"+pdf
    os.mkdir(nombreDir)
    call(["/home/claudia/PycharmProjects/similitudImagenes/pdfimages", "-j",nombretrabajo, nombreDir])
    #se eliminan posibles archivos que no son imagenes
    for img_path in ayudaDirectorios.getAllFilesInDirectory(nombreDir):
        print(img_path)
        if ".ppm" in img_path or ".pgm" in img_path:
            print("entro con la imagen " + img_path)
            os.remove(img_path)


#1.Se obtienen los subdirectorios de el Directorio general "directorios"
directorios = ayudaDirectorios.obtenerDirectorios()

#2. Se recorre cada subdirectorio y se añade el path de cada imagen a la lista imagenes
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
flags=[1,1,1,0,1,0,1,1,1,0,1,0]
if (len(flags) !=12):
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
maximos=[]
if(opcion=="a"): maximos=rangosPrefijados(flags)
elif(opcion=="c"):
    maximos=rangosImagenesAleatorias(flags, directorios)
if(opcion!="b" and opcion!="a" and opcion!="c"):
    raise AssertionError()

#5. Tras calcular los rangos de similitud por comparador se procede a comparar cada imagen de un directorio con las demás
# imágenes dando como resultado un diccionario con las posibles copias
# tambien se genera un archivo txt con un informe de los posibles plagios
diccionarioGlobal=calculaDiccionario(imagenes, flags, maximos, opcion)
ayudaDirectorios.pretty(diccionarioGlobal)



