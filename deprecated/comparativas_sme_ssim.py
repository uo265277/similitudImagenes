# import the necessary packages
#from skimage.measure import structural_similarity as ssim
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2


def escalaGrises(img_path):
    # Reading the image from the present directory
    image = cv2.imread(img_path)
    # Resizing the image for compatibility
    image = cv2.resize(image, (224, 224))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(img_path, image_bw)
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()

#
# # load the images -- the original, the original + contrast,
# # and the original + photoshop
# print(1)
# path1=r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\CLAUDIA\Claudia-038.jpg"
# path2=r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\RAQUEL\Raquel-030.jpg"
# path3=r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\CLAUDIA\Claudia-000.jpg"
# path4=r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\RAQUEL\Raquel-002.jpg"
# im1 = cv2.imread(path1)
# im2 = cv2.imread(path2)
# im3 = cv2.imread(path3)
# im4 = cv2.imread(path4)
#
# # convert the images to grayscale
# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# im1=cv2.resize(im1, (224, 224))
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# im2=cv2.resize(im2, (224, 224))
# im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
# im3=cv2.resize(im3, (224, 224))
# im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
# im4=cv2.resize(im4, (224, 224))
#
# # initialize the figure
# fig = plt.figure("Images")
# images = ("im1", im1), ("im2", im2), ("im3", im3) , ("im4", im4)
# # loop over the images
# for (i, (name, image)) in enumerate(images):
# 	# show the image
# 	ax = fig.add_subplot(1, 4, i + 1)
# 	ax.set_title(name)
# 	plt.imshow(image, cmap = plt.cm.gray)
# 	plt.axis("off")
# # show the figure
# plt.show()
# # compare the images
# compare_images(im1, im2, "im1 vs. im2")
# compare_images(im2, im3, "im2 vs. im3")
# compare_images(im3, im1, "im3 vs. im1")
# compare_images(im4, im1, "im4 vs. im1")
# compare_images(im4, im3, "im4 vs. im3")
# compare_images(im2, im4, "im2 vs. im4")
