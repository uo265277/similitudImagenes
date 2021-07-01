import os, sys

# obtiene el directorio dir* de una imagen dado su path
from os import listdir
from os.path import isfile, join


def directorioImagen(imagen):
    # -2 porque es el tercer subdirectorio: /content/pruebas_directorio/dir2/bear.jpg
    directorio =imagen.split('/')[-2]
    return directorio


# *******************************************************************************

# obtiene el nombre de una imagen dado su path
def nombreImagen(imagen):
    # -1 porque es lo ultimo a obtener: /content/pruebas_directorio/dir2/bear.jpg
    nombre =imagen.split('/')[-1]
    return nombre


# *******************************************************************************
# metodo que dado un path de una imagen devuelve el path pero sin la imagen

def pathSinImagen(path):

    # -2 porque es el tercer subdirectorio: /content/pruebas_directorio/dir2/bear.jpg
    imagen =path.split('/')[-1]

    solodir =path.replace(imagen, "")
    return solodir


# *******************************************************************************


# no se usa de momento
def vectorDirectorios(feature_vectors):
    # x="dir1/imagen.jpeg"
    # imagen=x.split('/')[-1]
    # directorio=x.split('/')[0]
    # print(imagen)
    # print(directorio)
    vDir =[]
    for k in feature_vectors:
        directorio =k.split('/')[0]
        vDir.append(directorio)
    return vDir


# *******************************************************************************
# obtiene los subdirectorios de la carpeta pruebas_directorio
def obtenerDirectorios():
    contenido = os.listdir('directorios')
    return contenido

def obtenerTrabajos():
    contenido = os.listdir('pdf')
    return contenido
# *******************************************************************************
# imprime un diccionari de una forma más visual
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent +1)
        else:
            print('\t' * (indent +1) + str(value))


# *******************************************************************************
# devuelve el numero de imagenes que hay en un directorio

# cuenta archivos, deben ser todos imagenes, no se como controlarlo hay que modificar etso
def contImagenes(dir):
    import os
    list = os.listdir(dir) # dir el path de lo que se quiere contar
    number_files = len(list)
    return number_files

#************************************************************************************
from PIL import Image
def pasarAJPG(path):
    im = Image.open(path)
    nombreFinal=path.split(".")[0]+".jpg"
    im.save(nombreFinal)
    os.remove(path)

#***********************************************************


def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]