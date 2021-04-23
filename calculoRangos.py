# 0 escalagrises
# 1 normalizado
# 2 clahe
# 3 hog
# 4 gabor
# 5 sift_sim
# 6 ssim
# 7 mse
# 8 gabor_sift_sim
import cv2

from datasetModificado.modificaImagen import recortar, escalaGrises, modificaColor
from normalizacionComparadores import obtieneMaxComparadores
from pruebaComparativas import comparaImagenes


def calculoRangos(path, flags):
    #obtengo una serie de modificaciones para la imagen
    #blanco y negro
    #recorte
    #filtro (modificacion de un canal de color)
    original = cv2.imread(path)
    #obtengo la lista de resultados de comparadores
    imgRecortada=recortar(path)
    pathRecortada="calculoRangos/imgRecortada.jpg"
    cv2.imwrite(pathRecortada, imgRecortada)

    imgByN=escalaGrises(path)
    pathByN = "calculoRangos/imgByN.jpg"
    cv2.imwrite(pathByN, imgByN)

    imgColorMod=modificaColor(path, 150, "azul")
    pathColorMod = "calculoRangos/imgColorMod.jpg"
    cv2.imwrite(pathColorMod, imgColorMod)

    #el rango para cada comparador es [ 0, max(resultado aproximado a 2 decimales)]
    resultRecortada=comparaImagenes(path,pathRecortada,flags)
    resultByN=comparaImagenes(path,pathByN,flags)
    resultColorMod=comparaImagenes(path,pathColorMod,flags)

    resultados=[ resultRecortada,resultByN,resultColorMod]
    maximos=obtieneMaxComparadores(resultados, flags)


    return maximos
