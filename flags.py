import cv2
from tensorflow import keras
import numpy as np
import skimage
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from skimage import measure
from skimage import feature
import math
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image


def predice(img, model: Model):
    #print("llego a predice")
    x=cv2.merge([img,img,img])
    #x = image.img_to_array(img)
    #print("traza1 "+ str(x.shape))
    x = np.expand_dims(x, axis=0)
    #print("traza2 "+ str(x.shape))
    x = preprocess_input(x)
    #print("traza3 "+ str(x.shape))
    return model.predict(x)

def findDifference(f1, f2):
    return np.linalg.norm(f1-f2)

model = ResNet50(weights='imagenet')

#FUNCION QUE PREPROCESA CON ESCALA DE GRISES PARA DESPUES CALCULAR SIMILITUD COMPARANDO AMBOS VECTORES DE CARACTERÍSTICAS
def escalaGrises(img_path, img_path2):
    print("estoy en escala grises")
    global  model
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))
    image2 = cv2.imread(img_path2)
    # Resizing the image for compatibility
    image2 = cv2.resize(image2, (224, 224))
    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #print("ya pase a gris " )
    diff =findDifference(predice(image_bw,model) ,predice(image2_bw,model))
    #print( diff)
    return diff

#FUNCION QUE PREPROCESA CON NORMALIZADO PARA DESPUES CALCULAR SIMILITUD COMPARANDO AMBOS VECTORES DE CARACTERÍSTICAS
def normalizado(img_path, img_path2):
    print("estoy en normalizado")
    global model
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))
    image2 = cv2.imread(img_path2)
    # Resizing the image for compatibility
    image2 = cv2.resize(image2, (224, 224))
    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.normalize(image_bw, image_bw, alpha=20, beta=200, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(image2_bw, image2_bw, alpha=20, beta=200, norm_type=cv2.NORM_MINMAX)
    diff = findDifference(predice(image_bw, model), predice(image2_bw, model))
    return diff

#FUNCION QUE PREPROCESA CON CLAHE PARA DESPUES CALCULAR SIMILITUD COMPARANDO AMBOS VECTORES DE CARACTERÍSTICAS
def clahe(img_path, img_path2):
    print("estoy en clahe")
    global model
    image = cv2.imread(img_path)
    image2 = cv2.imread(img_path2)
    image = cv2.resize(image, (224, 224))
    image2 = cv2.resize(image2, (224, 224))
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    image_bw = clahe.apply(image_bw) + 30
    image2_bw = clahe.apply(image2_bw) + 30
    diff = findDifference(predice(image_bw, model), predice(image2_bw, model))
    return diff

#FUNCION QUE PREPROCESA CON HOG PARA DESPUES CALCULAR SIMILITUD COMPARANDO AMBOS VECTORES DE CARACTERÍSTICAS
def hog(img_path, img_path2):
    print("estoy en hog")
    global model
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    # resizing image
    resized_img = cv2.resize(img, (128 * 4, 64 * 4))
    resized_img2 = cv2.resize(img2, (128 * 4, 64 * 4))
    # creating hog features
    fd, hog_image = skimage.feature.hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd, hog_image2 = skimage.feature.hog(resized_img2,  orientations=9,pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    diff = findDifference(predice(hog_image, model), predice(hog_image2, model))
    return diff

#FUNCION QUE PREPROCESA CON GABOR PARA DESPUES CALCULAR SIMILITUD COMPARANDO AMBOS VECTORES DE CARACTERÍSTICAS
def gabor(img_path, img_path2):
    print("estoy en gabor")
    global model
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, 4 * np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img2 = cv2.filter2D(img2, cv2.CV_8UC3, g_kernel)
    filtered_img = cv2.resize(filtered_img, (224, 224))
    filtered_img2 = cv2.resize(filtered_img2, (224, 224))
    diff = findDifference(predice(filtered_img, model), predice(filtered_img2, model))
    return diff

#FUNCION QUE APLICA SIFT PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def sift_sim(path_a, path_b):
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
    return 1
  #print(desc_b)
  #print(desc_a)
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 60]
  resultado = len(similar_regions) / len(matches)
  if len(matches) < 16:
    return 1
  return 1-resultado

#FUNCION QUE APLICA SSIM PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def ssim(img_path, img_path2):
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    image = cv2.resize(img, (224, 224))
    image2 = cv2.resize(img2, (224, 224))
    #multichannel a true ya que es imagen a color
    s = measure.compare_ssim(image, image2, multichannel=True)
    return 1-s


#FUNCION QUE APLICA MSE PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def mse(img_path, img_path2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    imageA = cv2.imread(img_path)
    imageB = cv2.imread(img_path2)
    imageA = cv2.resize(imageA, (224, 224))
    imageB = cv2.resize(imageB, (224, 224))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err/100000



#FUNCION QUE APLICA GABOR+SIFT PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def gabor_sift_sim(img_path, img_path2):
    print("estoy en gabor + sift sim")
    orb = cv2.ORB_create()
    model = ResNet50(weights='imagenet')
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, 4 * np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img2 = cv2.filter2D(img2, cv2.CV_8UC3, g_kernel)
    filtered_img = cv2.resize(filtered_img, (224, 224))
    filtered_img2 = cv2.resize(filtered_img2, (224, 224))
    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(filtered_img, None)
    kp_b, desc_b = orb.detectAndCompute(filtered_img2, None)
    # print(desc_b  )
    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # match.distance is a float between {0:100} - lower means more similar
    #print(str(type(desc_a)) + str(type(desc_b)))
    if (str(type(desc_b)) == "<class 'NoneType'>" or str(type(desc_a)) == "<class 'NoneType'>"):
        return 1
    #print(desc_b)
    #print(desc_a)
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 60]
    resultado = len(similar_regions) / len(matches)
    if len(matches) < 16:
        return 1
    return 1-resultado

#FUNCION QUE APLICA PSNR PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def psnr(img_path, img_path2):
    original = cv2.imread(img_path)
    compressed = cv2.imread(img_path2)
    original = cv2.resize(original, (224, 224))
    compressed = cv2.resize(compressed, (224, 224))
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 0
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return 1-(psnr/255)



#FUNCION QUE CALCULA LA FIRMA DE UNA IMAGEN
def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """
    Calculate the dhash signature of a given file

    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2

    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    try:
        pil_image = Image.open(image_file).convert("L").resize(
            (hash_size + 1, hash_size),
            Image.ANTIALIAS)
        dhash = imagehash.dhash(pil_image, hash_size)
        signature = dhash.hash.flatten()
        pil_image.close()
        return signature
    except IOError as e:
        raise e




#FUNCION QUE APLICA LSH PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def lsh(img_path1,img_path2):
    threshold = 0.0
    hash_size = 8
    bands = 10

    rows: int = int(hash_size ** 2 / bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]

    #calculo la firma de la primera imagen
    signature = calculate_signature(img_path1, hash_size)
    print("calculo la firma de "+img_path1)

    # Keep track of each image's signature
    signatures[img_path1] = np.packbits(signature)

    # Locality Sensitive Hashing
    for i in range(bands):
            signature_band = signature[i * rows:(i + 1) * rows]
            signature_band_bytes = signature_band.tostring()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(img_path1)

    signature2 = calculate_signature(img_path2, hash_size)
    print("calculo la firma de " + img_path2)

    # Keep track of each image's signature
    signatures[img_path2] = np.packbits(signature2)

    # Locality Sensitive Hashing
    for i in range(bands):
        signature_band = signature2[i * rows:(i + 1) * rows]
        signature_band_bytes = signature_band.tostring()
        if signature_band_bytes not in hash_buckets_list[i]:
            hash_buckets_list[i][signature_band_bytes] = list()
        hash_buckets_list[i][signature_band_bytes].append(img_path2)

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i + 1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i], hash_bucket[j]])
                        )
    # Check candidate pairs for similarity
    near_duplicates = list()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
            np.unpackbits(signatures[cpa]),
            np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size ** 2 - hd) / hash_size ** 2
        if similarity > threshold:
            near_duplicates.append((cpa, cpb, 1-similarity))
    near_duplicates.sort(key=lambda x: x[2], reverse=True)
    if (len(near_duplicates) != 0):
       return near_duplicates[0][2]
    else:
        return 1



#FUNCION QUE APLICA HISTOGRAMA DE COLOR PARA DETERMINAR LA SIMILITUD ENTRE DOS IMAGENES
def histogramaColor(img_path, img_path2):
    original = img_path
    comparada = img_path2
    image1 = cv2.imread(comparada)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(original)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image_list = [image1, image2]
    histograms = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in image_list]
    histn = [cv2.normalize(hist, hist).flatten() for hist in histograms]
    result = cv2.compareHist(histn[0], histn[1], cv2.HISTCMP_CORREL)
    result = 1 - result
    return result
