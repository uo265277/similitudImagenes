import cv2
from matplotlib import pyplot

import datasetModificado.modificaImagen
import flags
import numpy as np
from datasetModificado.modificaImagen import escalaGrises

path1=r"/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/B.jpg"
path2=r"/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/Brecortado.jpg"

# print(flags.escalaGrises(path1, path2))
# print(flags.normalizado(path1, path2))
# print(flags.clahe(path1, path2))
# print(flags.gabor(path1, path2))
# print(flags.clahe(path1, path2))
# print(flags.mse(path1, path2))
# print(flags.ssim(path1, path2))
# print(flags.sift_sim(path1, path2))


# a and b are numpy arrays
def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

from scipy import misc
from matplotlib import pyplot
import numpy as np

#Get two images with snippet at different locations

#print("phase correlation")
img1=cv2.imread(path1)
img1 = cv2.resize(img1, (224, 224))

img2=cv2.imread(path2)
img2 = cv2.resize(img2, (224, 224))

corrimg = phase_correlation(img1, img2)
r,c , _= np.unravel_index(corrimg.argmax(), corrimg.shape)

pyplot.imshow(img1)
pyplot.plot([c],[r],'ro')
pyplot.show()

pyplot.imshow(img2)
pyplot.show()

#pyplot.figure(figsize=[8,8])
pyplot.imshow(corrimg, cmap='gray')

pyplot.show()
#Obtain normalized cross-correlation by applying the inverse Fourier transform r = F^-1{R}
#Determine the location of the peak in r:
#(del_x, del_y) = argmax over (x,y) of {r}
