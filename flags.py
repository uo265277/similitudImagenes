import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from skimage import measure


def predice(img, model: Model):
    print("llego a predice")
    x=cv2.merge([img,img,img])
    #x = image.img_to_array(img)
    print("traza1 "+ str(x.shape))
    x = np.expand_dims(x, axis=0)
    print("traza2 "+ str(x.shape))
    x = preprocess_input(x)
    print("traza3 "+ str(x.shape))
    return model.predict(x)

def findDifference(f1, f2):
    return np.linalg.norm(f1-f2)


def escalaGrises(img_path, img_path2):
    print("estoy en escala grises")
    model = ResNet50(weights='imagenet')

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
    print("ya pase a gris " )
    diff =findDifference(predice(image_bw,model) ,predice(image2_bw,model))
    print( diff)
    return diff

def normalizado(img_path, img_path2):
    print("estoy en normalizado")


    model = ResNet50(weights='imagenet')

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


def clahe(img_path, img_path2):
    print("estoy en clahe")

    model = ResNet50(weights='imagenet')

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

def hog(img_path, img_path2):
    print("estoy en hog")

    model = ResNet50(weights='imagenet')

    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    # resizing image
    resized_img = cv2.resize(img, (128 * 4, 64 * 4))
    resized_img2 = cv2.resize(img2, (128 * 4, 64 * 4))
    # creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd, hog_image2 = hog(resized_img2, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    diff = findDifference(predice(hog_image, model), predice(hog_image2, model))
    return diff

def gabor(img_path, img_path2):
    print("estoy en gabor")

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
    diff = findDifference(predice(filtered_img, model), predice(filtered_img2, model))
    return diff

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
    return 0
  print(desc_b)
  print(desc_a)
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 60]
  resultado = len(similar_regions) / len(matches)
  if len(matches) < 16:
    return 0
  return resultado


def ssim(img_path, img_path2):
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    image = cv2.resize(img, (224, 224))
    image2 = cv2.resize(img2, (224, 224))
    #multichannel a true ya que es imagen a color
    s = measure.compare_ssim(image, image2, multichannel=True)

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
    return err