# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

original="directorios/IMAGEN1/IMAGEN1_original.jpg"
comparada="directorios/IMAGEN1/IMAGEN1_rotadal.jpg"


image1 = cv2.imread(comparada)
image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread(original)
image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

image_list=[image1, image2]
histograms = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in image_list]
histn = [cv2.normalize(hist, hist).flatten() for hist in histograms]

result=cv2.compareHist(histn[0], histn[1], cv2.HISTCMP_CORREL)

result=1-result

print(result)