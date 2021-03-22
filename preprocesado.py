# When applying CLAHE, there are two parameters to be remembered:
# clipLimit – This parameter sets the threshold for contrast limiting. The default value is 40.
# tileGridSize – This sets the number of tiles in the row and column. By default this is 8×8. It is used while the image is divided into tiles for applying CLAHE.
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

def aplicaCLAHE(img_path: str):
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting

    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30

    # Ordinary thresholding the same image
    # _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

    # pathDondeMeterImagenNueva= pathSinImagen(img_path):
    # nombreImagen = nombreImagen(img_path)

    cv2.imwrite(img_path, final_img)
    ##############cv2.imwrite(img_path,image_bw)


# Showing all the three images
# cv2.imshow("ordinary threshold", ordinary_img)
# cv2.imshow("CLAHE image", final_img)


# *******************************************************************************
def escalaGrises(img_path):
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(img_path, image_bw)


# *******************************************************************************

def normalizaImagen(img_path):
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    cv2.normalize(image, image, alpha=20, beta=200, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(img_path, image)


# *******************************************************************************

def bordeImagen(img_path):
    img = cv2.imread(img_path)

    edge_img = cv2.Canny(img, 100, 200)

    cv2.imwrite(img_path, edge_img)


# *******************************************************************************


def contornoImagen(img_path):
    img = cv2.imread(img_path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # buscar umbral
    retval, thresh = cv2.threshold(gray_img, 127, 255, 0)
    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, img_contours, -1, (0, 255, 0))
    cv2.imwrite(img_path, img)


# *******************************************************************************

def gabor(img_path):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, 4 * np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    cv2.imwrite(img_path, filtered_img)


# *******************************************************************************

def aplicar_hog(img_path):
    img = cv2.imread(img_path)
    plt.axis("off")
    plt.imshow(img)

    print(img.shape)

    # resizing image
    resized_img = cv2.resize(img, (128 * 4, 64 * 4))
    plt.axis("off")
    # plt.imshow(resized_img)
    # print(resized_img.shape)

    # creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    # plt.axis("off")
    # plt.imshow(hog_image, cmap="gray")
    # save the images
    # plt.imsave("resized_img.jpg", resized_img)
    plt.imsave(img_path, hog_image, cmap="gray")


#escalaGrises(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
#aplicaCLAHE(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
#normalizaImagen(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
#bordeImagen(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
#contornoImagen(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
#gabor(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
aplicar_hog(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\coati.jpg")
