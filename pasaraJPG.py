
import os
from PIL import Image
from os.path import isfile, join

def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in os.listdir(directoryPath) if isfile(join(directoryPath, f))]

def pasarAJPG(path):
    im = Image.open(path)
    nombreFinal=path.split(".")[0]+".jpg"
    im.save(nombreFinal)
    os.remove(path)
    return nombreFinal




