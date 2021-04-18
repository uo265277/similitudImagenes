from keras.applications.resnet50 import ResNet50
import ayudaDirectorios
import diccionarioDistancias
import preprocesado
from subprocess import call

feature_vectors: dict = {}
model = ResNet50(weights='imagenet')
imagenes: dict = {}

#call(["pdfimages.exe", "-j", "pdf","directorios\\mapaches"])


# model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# ¿como saco las imagenes de cada pdf y las añado a cada directorio? -> pdfimages para texto pdftxt
# adaptar para que se compare cada imagen de dir x con cada imagen de los diferentes directorios
directorios = ayudaDirectorios.obtenerDirectorios()
cont  =0
for directorio in directorios:
    print("recorro el directorio "+directorio)
    for img_path in ayudaDirectorios.getAllFilesInDirectory("directorios/"+directorio):
        #nombreFinal=ayudaDirectorios.pasarAJPG(img_path)
        cont=cont+1
        imagenes[img_path] = cont
        #print(img_path)
        feature_vectors[img_path] = preprocesado.predict(img_path, model)[0]
# results=findDifferences(feature_vectors)
ayudaDirectorios.pretty(imagenes)

print("****************************************************************************************")

# diccionarioGlobal=calculaDiccionarioDistancia3(feature_vectors)
#diccionarioGlobal = diccionarioDistancias.calculaDiccionarioDistancia4(feature_vectors,0.02)
diccionarioGlobal = diccionarioDistancias.calculaDiccionarioDistancia4(feature_vectors, 0.1
                                                                       )
print()
print()
print()
print()
print("procedo a imprimir la estructura de diccionario:")
print()

print("Claves de el Diccionario global, siendo cada clave un diccionario por directorio", diccionarioGlobal.keys())
print("Valores de el Diccionario global, siendo valor el diccionario por directorio", diccionarioGlobal.values())

print("****************************************************************************************")

ayudaDirectorios.pretty(diccionarioGlobal)