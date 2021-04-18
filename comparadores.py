import warnings
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from imageio import imread
import numpy as np
import cv2

#https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c

##
# Globals
##


import ayudaDirectorios

def calculaDiccionarioDistancia6(imagenes, ratio):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont = 0
    dirAux = ""

    for k in imagenes:

      # CREO EL DICCIONARIO DE DIRECTORIO
      dirK = ayudaDirectorios.directorioImagen(k)

      # si estoy en el mismo directorio
      if cont == 0:
        print("traza  if cont==0: ")
        cont = cont + 1
        dirAux = dirK
        diccionarioDirectorios = "DiccionarioDelDirectorio " + dirK
        print(diccionarioDirectorios)
        diccionarioDirectorios: dict = {}

        # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
        diccionarioGlobal[dirK] = diccionarioDirectorios

      print()
      print()
      print()
      print()
      print()
      print("*************DICCIONARIO DE DIRECTORIO  " + dirK + " ******************")
      # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

      # si el contador es 0 quiere decir que he cambiado de directorio
      if (dirAux != dirK):
        print("traza if (dirAux!=dirK): ")
        diccionarioDirectorios = "DiccionarioDelDirectorio " + dirK
        print(diccionarioDirectorios)
        diccionarioDirectorios: dict = {}
        dirAux = dirK

        # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
        diccionarioGlobal[dirK] = diccionarioDirectorios

      # CREO EL SUBDICCIONARIO DE LA IMAGEN
      nomK = ayudaDirectorios.nombreImagen(k)
      diccionarioImagen = "SubDiccionarioImagen " + dirK + nomK
      print()
      print("--------------  " + diccionarioImagen + " -----------------------------")
      diccionarioImagen: dict = {}

      # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
      diccionarioDirectorios[dirK + nomK] = diccionarioImagen
      # print("Estado del diccionadio de directorios "+ dirK+" : ")
      # print(diccionarioDirectorios)
      # print("fin")
      for k2 in imagenes:
        dirK2 = ayudaDirectorios.directorioImagen(k2)
        nomK2 = ayudaDirectorios.nombreImagen(k2)
        if (dirK != dirK2):
          # saco los path absolutos porque isno da error con los relativos
          print("calculo la diff " + k + k2)
          sift_sim = pruebaSiftSim(k, k2)
          print("El sift sim de ", k, " con ", k2, "es", sift_sim)
          #diccionarioImagen[dirK2 + nomK2] = sift_sim
          if (sift_sim >= ratio):
                             diccionarioImagen[dirK2 +nomK2 ] =sift_sim
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



#############################################################################################################################################
warnings.filterwarnings('ignore')

# specify resized image sizes
height = 2**10
width = 2**10

##
# Functions
##

def get_img(path, norm_size=True, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # flatten returns a 2d grayscale array
  img = imread(path, as_gray=True).astype(int)

  # resizing returns float vals 0:255; convert to ints for downstream tasks
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img

def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)


def normalize_exposure(img):
  '''
  Normalize the exposure of an image.
  '''
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  print(path_a + path_b)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)
#############################################################################################################################################


#earth_movers_distance(pathraquel, pathclaudia)

def structural_sim(path_a, path_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = compare_ssim(img_a, img_b, full=True)
  return sim

def pixel_sim(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


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
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)
  print(desc_b  )
  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)

  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)









def pruebaSiftSim(path_a, path_b):
  print("path dentro pruebasSiftSim: "+ path_a +" " + path_b)
  orb = cv2.ORB_create()
  # get the images
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)
  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)
 # print(desc_b  )
  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # match.distance is a float between {0:100} - lower means more similar
  print(str(type(desc_a)) + str(type(desc_b)))

  if(str(type(desc_b))=="<class 'NoneType'>" or str(type(desc_a))=="<class 'NoneType'>"):
    return 0
  print(desc_b)
  print(desc_a)
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 60]
  resultado = len(similar_regions) / len(matches)
  if len(matches) < 16:
    return 0
  return resultado




import os
from os import listdir
from os.path import isfile, join


def getAllFilesInDirectory(directoryPath: str):
  return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]


if __name__ == '__main__':
  imagenes: dict = {}
  for path in getAllFilesInDirectory('directorios/CLAUDIA'):
    print(path)
    imagenes[path]=path
  for path in getAllFilesInDirectory('directorios/MARIA'):
    print(path)
    imagenes[path]=path
  # #img_a = os.path.abspath("datasetModificado\CLAUDIA\Claudia-001.jpg")
  # img_a= "directorios/RAQUEL/Raquel-002.jpg"
  # img_b= "directorios/CLAUDIA/Claudia-000.jpg"
  # # get the similarity values
  # structural_sim2 = structural_sim(img_a, img_b)
  # pixel_sim2 = pixel_sim(img_a, img_b)
  # sift_sim2 = sift_sim(img_a, img_b)
  # emd2 = earth_movers_distance(img_a, img_b)
  # print(structural_sim2, pixel_sim2, sift_sim2, emd2)
  #
  # k="directorios/RAQUEL/Raquel-000.jpg"
  # k2="directorios/RAQUEL/Raquel-000.jpg"
  #
  # sift_sim = pruebaSiftSim(img_a, img_b)
  # print("El sift sim de ", k, " con ", k2, "es", sift_sim)
  dir=calculaDiccionarioDistancia6(imagenes, 0.99)
  ayudaDirectorios.pretty(dir)

  print("empieza lo interesante con el diccionario")

