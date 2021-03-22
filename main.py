from keras.applications.resnet50 import ResNet50
import ayudaDirectorios
import diccionarioDistancias
import preprocesado

feature_vectors: dict = {}
model = ResNet50(weights='imagenet')
# model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# ¿como saco las imagenes de cada pdf y las añado a cada directorio? -> pdfimages para texto pdftxt
# adaptar para que se compare cada imagen de dir x con cada imagen de los diferentes directorios
directorios = ayudaDirectorios.obtenerDirectorios()
for directorio in directorios:
    for img_path in ayudaDirectorios.getAllFilesInDirectory("directorios\\"+directorio):
        feature_vectors[img_path] = preprocesado.predict(img_path, model)[0]
# results=findDifferences(feature_vectors)


print("****************************************************************************************")

# diccionarioGlobal=calculaDiccionarioDistancia3(feature_vectors)
diccionarioGlobal = diccionarioDistancias.calculaDiccionarioDistancia3(feature_vectors)
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