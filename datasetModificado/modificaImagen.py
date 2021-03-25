import cv2, numpy as np


def recortar (path):
    img = cv2.imread(path)
    alto = img.shape[0]
    ancho = img.shape[1]
    crop_img = img[int(alto/4):int(3*alto/4),int(ancho/4):int(3*ancho/4)]
    recortado="recortado"
    nombre=path.split(".")
    print(nombre[0])
    nuevopath=nombre[0]+recortado+".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, crop_img)



def rotar(path):
    original = cv2.imread(path)
    alto = original.shape[0]
    ancho = original.shape[1]
    M = cv2.getRotationMatrix2D((ancho / 2, alto / 2), 90, 1)
    M[0][2]=0
    M[1][2]=ancho
    new90center = cv2.warpAffine(original, M, (alto, ancho))
    rotado = "rotado"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + rotado + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, new90center)

#rotar(path)


def modificaColor(path, nivelRealce, color):
    original = cv2.imread(path)
    # dividir en bandas
    (blue, green, red) = cv2.split(original)
    # nivel de realce del rojo
    delta = nivelRealce

    if (color == "rojo"):

        # banda rojo incrementada
        redEnhanced = (((red <= (255 - delta)).astype(np.uint8)) * (red + delta)) + (
            (red > (255 - delta)).astype(np.uint8)) * 255
        # nueva imagen con rojo incrementado
        redEnhancedImage = cv2.merge([blue, green, redEnhanced])

        rojoModificado = "RojoModificado"
        nombre = path.split(".")
        print(nombre[0])
        nuevopath = nombre[0] + rojoModificado + ".jpg"
        print(nuevopath)
        cv2.imwrite(nuevopath, redEnhancedImage)
    if (color == "verde"):
        # banda verde incrementada
        greenEnhanced = (((green <= (255 - delta)).astype(np.uint8)) * (green + delta)) + (
            (green > (255 - delta)).astype(np.uint8)) * 255
        # nueva imagen con verde incrementado
        greenEnhancedImage = cv2.merge([blue, greenEnhanced, red])

        verdeModificado = "VerdeModificado"
        nombre = path.split(".")
        print(nombre[0])
        nuevopath = nombre[0] + verdeModificado + ".jpg"
        print(nuevopath)
        cv2.imwrite(nuevopath, greenEnhancedImage)
    if (color == "azul"):
        # banda azul incrementada
        blueEnhanced = (((blue <= (255 - delta)).astype(np.uint8)) * (blue + delta)) + (
            (blue > (255 - delta)).astype(np.uint8)) * 255
        # nueva imagen con azul incrementado
        blueEnhancedImage = cv2.merge([blueEnhanced, green, red])

        azulModificado = "AzulModificado"
        nombre = path.split(".")
        print(nombre[0])
        nuevopath = nombre[0] + azulModificado + ".jpg"
        print(nuevopath)
        cv2.imwrite(nuevopath, blueEnhancedImage)

#modificaColor(path, 40, "azul")
#modificaColor(path, 40, "verde")
#modificaColor(path, 40, "rojo")





def traslacion(path, ejex, ejey):
    original = cv2.imread(path)
    alto = original.shape[0]
    ancho = original.shape[1]

    tx = ejex
    ty = ejey
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(original, M, (ancho, alto))

    trasladado = "Trasladado"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + trasladado + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, translated)

#traslacion(path, 30, 50)



def zoomDiferencia(path):
    original = cv2.imread(path)
    alto = original.shape[0]
    ancho = original.shape[1]
    zoomedInDefault = cv2.resize(original, (int(ancho * 2), int(alto * 2)))
    zoomedInInterCubic = cv2.resize(original, (int(ancho * 2), int(alto * 2)), interpolation=cv2.INTER_CUBIC)
    diff = cv2.cvtColor(np.abs(zoomedInDefault - zoomedInInterCubic), cv2.COLOR_BGR2GRAY)
    zoomDiferencia= "ZoomDiferencia"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + zoomDiferencia + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, diff)

#zoomDiferencia(path, 2)



def reduccionRuido(path, fuerza,):
    img = cv2.imread(path)
    #imagen, destino, fuerza filtro, valor imagen ara eliminar ruido
    #(normalmente igual a la fuerza del filtro o 10),el tamaño del parche
    # #de la plantilla en píxeles para calcular los pesos que siempre deberían
    # #ser impares (el tamaño recomendado es igual a 7) y el tamaño de la ventana
    # en píxeles para calcular el promedio del píxel dado.
    result = cv2.fastNlMeansDenoisingColored(img, None, fuerza, 10, 7, 21)
    reduccionRuido= "ReduccionRuido"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + reduccionRuido + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, result)

#reduccionRuido(path, 30)

def escalaGrises(path):
    image = cv2.imread(path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    escalaGrises= "EscalaGrises"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + escalaGrises + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, image_bw)

#escalaGrises(path)



def desenfoqueMediano(path,blur):
    img = cv2.imread(path)

    blur_image = cv2.medianBlur(img, blur)


    desenfoqueMediano = "DesenfoqueMediano"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + desenfoqueMediano + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, blur_image)

#desenfoqueMediano(path, 7)



def ajusteContraste(path, contraste):
    img = cv2.imread(path)

    contrast_img = cv2.addWeighted(img, contraste, np.zeros(img.shape, img.dtype), 0, 0)

    ajusteContraste = "AjusteContraste"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + ajusteContraste + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, contrast_img)

#ajusteContraste(path, 3)

def sinModificar(path):
    img = cv2.imread(path)
    sinModificar = "SinModificar"
    nombre = path.split(".")
    print(nombre[0])
    nuevopath = nombre[0] + sinModificar + ".jpg"
    print(nuevopath)
    cv2.imwrite(nuevopath, img)





