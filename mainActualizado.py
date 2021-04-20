import ayudaDirectorios
from diccionarioGlobal import calculaDiccionario
from normalizacionComparadores import obtieneMaxComparadores



#Se obtienen los subdirectorios de el Directorio general "directorios"
directorios = ayudaDirectorios.obtenerDirectorios()

#se recorre cada subdirectorio y se añade -> PATH_IMAGEN
imagenes=[]
for directorio in directorios:
    print("recorro el directorio "+directorio)
    for img_path in ayudaDirectorios.getAllFilesInDirectory("directorios/"+directorio):
        imagenes.append(img_path)


diccionarioGlobal=calculaDiccionario(imagenes, [1,1,1,1,1,1,0,1,1])

#a este diccionario global hay que sacarle todas las listas de subdiccionarios y fusionarlos en una lista
#para sacar el maximo

#obtieneMaxComparadores(lista,  [1,1,1,1,1,1,0,1,1])


ayudaDirectorios.pretty(diccionarioGlobal)