import cv2

import datasetModificado.modificaImagen
import flags
from datasetModificado.modificaImagen import escalaGrises

path1=r"/home/claudia/PycharmProjects/similitudImagenes/directorios/CLAUDIA_PRUEBAS/D.jpg"
imgByN=datasetModificado.modificaImagen.escalaGrises(path1 )
path2 = "calculoRangos/imgByN.jpg"
cv2.imwrite(path2, imgByN)
path2=r"/home/claudia/PycharmProjects/similitudImagenes/directorios/CLAUDIA_PRUEBAS/B.jpg"

print(flags.escalaGrises(path1, path2))
print(flags.normalizado(path1, path2))
print(flags.clahe(path1, path2))
print(flags.gabor(path1, path2))
print(flags.clahe(path1, path2))
print(flags.mse(path1, path2))
print(flags.ssim(path1, path2))
print(flags.sift_sim(path1, path2))




#CONCLUSIONES
#ENTRE IMAGENES IGUALES TODOS VAN BIEN
#ENTRE IMAGENES CON CAMBIOS EN COLOR O EN BYN todos van bien sobretodo escalagrises, normalizado, clahe, gabor, mse, ssim y sift sim
#entre imagenes recortadas no sirven -> probar con phase correlation