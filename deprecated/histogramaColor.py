# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

#original="datasetModificado/homer.jpg"
#comparada="datasetModificado/homerRotado.jpg"


# image1 = cv2.imread(comparada)
# image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image2 = cv2.imread(original)
# image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#
# image_list=[image1, image2]
# histograms = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in image_list]
# histn = [cv2.normalize(hist, hist).flatten() for hist in histograms]
#
# result=cv2.compareHist(histn[0], histn[1], cv2.HISTCMP_CORREL)
#
# result=1-result


import matplotlib.pyplot as plt
import cv2
import os


img = cv2.imread("datasetModificado/homerRotado.jpg")    # Load the image
channels = cv2.split(img)       # Set the image channels
colors = ("b", "g", "r")        # Initialize tuple
plt.figure()
plt.title("Histograma de color rotado")
plt.xlabel("Bins")
plt.ylabel("Numero de pixeles")

for (i, col) in zip(channels, colors):       # Loop over the image channels
          hist = cv2.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
          plt.plot(hist, color = col)      # Plot the histogram
          plt.xlim([0, 256])

plt.show()