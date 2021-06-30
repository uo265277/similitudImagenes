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
#9 psnr
#10 lsh
#11 histograma color
import random
import cv2

import ayudaDirectorios
from datasetModificado.modificaImagen import recortar, escalaGrises, modificaColor, recorte
from normalizacionComparadores import obtieneMaxComparadores
from pruebaComparativas import comparaImagenes


def calcularRangos(path, flags, tolerancia):
    #obtengo una serie de modificaciones para la imagen
    #y obtengo la lista de resultados de comparadores para cad modificacion

    imgByN=escalaGrises(path)
    pathByN = "calculoRangos/imgByN.jpg"
    cv2.imwrite(pathByN, imgByN)

    imgColorMod=modificaColor(path, 150, "azul")
    pathColorMod = "calculoRangos/imgColorMod.jpg"
    cv2.imwrite(pathColorMod, imgColorMod)

    porcentajeMinimoCrop = 0.7
    imgRecortada=recorte(path)
    pathRecortada= "calculoRangos/imgRecortada.jpg"
    cv2.imwrite(pathRecortada, imgRecortada)


    #el rango para cada comparador es [ 0, max(resultado + tolerancia aproximado a 2 decimales)]
    #resultRecortada=comparaImagenes(path,pathRecortada,flags)
    resultByN=comparaImagenes(path,pathByN,flags)
    resultColorMod=comparaImagenes(path,pathColorMod,flags)
    resultRecortada=comparaImagenes(path,pathRecortada,flags)


    resultados=[resultByN,resultColorMod,resultRecortada]
    maximos=obtieneMaxComparadores(resultados, flags)
    for max in maximos:
        max=round(max+tolerancia, 2)

    return maximos


#Se devuelve una lista con sublistas tal como [[metodo, maximo],...]
def rangosPrefijados(flags):
    #parámetros predeterminados para rangos prefijados
    escalagrises=0.1
    normalizado=0.1
    clahe=0.1
    hog=0.1
    gabor=0.1
    sift_sim=0.1
    ssim=0.1
    mse=0.1
    gabor_sift_sim=0.1
    psnr=0.1
    lsh=0.1
    histograma_color=0.1

    cont=0
    maximos=[]
    #para cada flag si es 1 añado el maximo, sino no
    if(flags[0]==1):
        metodo= "escalaGrises"
        cadena = [metodo, escalagrises]
        maximos.append(cadena)
    if(flags[1]==1):
        metodo= "normalizado"
        cadena = [metodo, normalizado]
        maximos.append(cadena)
    if(flags[2]==1):
        metodo= "clahe"
        cadena = [metodo, clahe]
        maximos.append(cadena)
    if(flags[3]==1):
        metodo= "hog"
        cadena = [metodo, hog]
        maximos.append(cadena)
    if(flags[4]==1):
        metodo= "gabor"
        cadena = [metodo, gabor]
        maximos.append(cadena)
    if(flags[5]==1):
        metodo= "sift_sim"
        cadena = [metodo, sift_sim]
        maximos.append(cadena)
    if(flags[6]==1):
        metodo= "ssim"
        cadena = [metodo, ssim]
        maximos.append(cadena)
    if(flags[7]==1):
        metodo= "mse"
        cadena = [metodo, mse]
        maximos.append(cadena)
    if(flags[8]==1):
        metodo= "gabor_sift_sim"
        cadena = [metodo, gabor_sift_sim]
        maximos.append(cadena)
    if(flags[9]==1):
        metodo= "psnr"
        cadena = [metodo, psnr]
        maximos.append(cadena)
    if(flags[10]==1):
        metodo= "lsh"
        cadena = [metodo, lsh]
        maximos.append(cadena)
    if(flags[11]==1):
        metodo= "histogramaColor"
        cadena = [metodo, histograma_color]
        maximos.append(cadena)
    return maximos














def rangosImagenesIndividuales(flags, path):
    imgByN = escalaGrises(path)
    pathByN = "calculoRangos/imgByN.jpg"
    cv2.imwrite(pathByN, imgByN)

    imgColorMod = modificaColor(path, 150, "azul")
    pathColorMod = "calculoRangos/imgColorMod.jpg"
    cv2.imwrite(pathColorMod, imgColorMod)

    # porcentajeMinimoCrop = 0.7
    # imgRecortada = recorte(path, porcentajeMinimoCrop, porcentajeMinimoCrop)
    # pathRecortada = "calculoRangos/imgRecortada.jpg"
    # cv2.imwrite(pathRecortada, imgRecortada)

    # el rango para cada comparador es [ 0, max(resultado + tolerancia aproximado a 2 decimales)]
    # resultRecortada=comparaImagenes(path,pathRecortada,flags)
    tolerancia=0.000
    resultByN = comparaImagenes(path, pathByN, flags)
    resultColorMod = comparaImagenes(path, pathColorMod, flags)
    # resultRecortada = comparaImagenes(path, pathRecortada, flags)

    resultados = [resultByN, resultColorMod]
    maximos = obtieneMax(resultados, flags)
    return maximos



def obtieneMax(resultados,flags):
    longListaMax=flags.count(1)
    rangosMax=[]
    i=0
    while(i< longListaMax):
            if (resultados[0][i][1] > resultados[1][i][1]):
                rangosMax.append(resultados[0][i])
            else:
                rangosMax.append(resultados[1][i])
            i=i+1
    return rangosMax
def rangosImagenesAleatorias(flags,directorios):
    muestras=3
    resultados=[]
    x=0
    while(x<muestras):

        dirAleatorio = random.choices(directorios)
        dirAleatorio = dirAleatorio.pop()
        pathAleatorio = random.choices(ayudaDirectorios.getAllFilesInDirectory("directorios/" + dirAleatorio))
        pathAleatorio = pathAleatorio.pop()
        resultado=rangosImagenesIndividuales(flags,pathAleatorio)
        resultados.append(resultado)
        x=x+1
    #resultados = [  [[metodo,valor],[metodo,valor]...]...]
    longListaMax = flags.count(1)
    rangosMax = []
    i = 0
    while (i < longListaMax):
        #tomando 3 muestras, esto se puede optimizar
        media=resultados[0][i][1] + resultados[1][i][1] + resultados[2][i][1]
        media=media/3
        resultado=[resultados[0][i][0],media]
        rangosMax.append(resultado)
        i = i + 1
    return rangosMax








def estaEnRango(resultados, maximos):
    lista1=[]
    for sublista in resultados:
        comparador=sublista[0]
        valor=sublista[1]
        lista1.append(valor)
    lista2=[]
    for sublista in maximos:
        comparador=sublista[0]
        valor=sublista[1]
        lista2.append(valor)
    i=0
    pertenece=False
    while(i<len(lista2)):
        if (lista1[i]<lista2[i] or lista1[i]==lista2[i]):
            pertenece=True
        i=i+1
    return pertenece

