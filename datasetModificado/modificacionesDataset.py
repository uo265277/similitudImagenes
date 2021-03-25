import  random
import ayudaDirectorios
# dadas las imágenes de un directorio se les aplicarán
# modificaciones aleatorias
from datasetModificado.modificaImagen import recortar, rotar, modificaColor, traslacion, zoomDiferencia, reduccionRuido, \
    escalaGrises, desenfoqueMediano, ajusteContraste, sinModificar


def modificacionesAleatorias(directorio):
  #recorro imagenes de directorio
  # para cada imagen
  for img_path in ayudaDirectorios.getAllFilesInDirectory(directorio):
     print(img_path)
     #genero numero aleatorio entre 0 y 11 porque hay 11 tipos de modificacion + 1 no hacer nada
     mod= random.randint(0, 11)
     print(mod)

     #le aplico esa modificacion
     if mod==0:
        recortar(img_path)

     elif mod==1:
       rotar(img_path)

     elif mod == 2:
      nivel= random.randint(30, 200)
      modificaColor(img_path, nivel, "rojo")

     elif mod == 3:
      nivel= random.randint(30, 200)
      modificaColor(img_path,nivel, "verde")

     elif mod == 4:
      nivel= random.randint(30, 200)
      modificaColor(img_path,nivel, "azul")

     elif mod == 5:
      ejex= random.randint(0, 200)
      ejey= random.randint(0, 200)
      traslacion(img_path, ejex, ejey)

     elif mod == 6:
      factor= random.randint(0, 3)
      zoomDiferencia(img_path)


     elif mod == 7:
      fuerza= random.randint(10, 50)
      reduccionRuido(img_path,fuerza)

     elif mod == 8:
      escalaGrises(img_path)


     elif mod == 9:
     #numeros mayor que 1 e impar
      odd_list = [3,5,7,9,11,13,15]
      blur=random.choice(odd_list)
      desenfoqueMediano(img_path, blur)

     elif mod == 10:
      contraste= random.randint(0, 3)
      ajusteContraste(img_path, contraste)

     else:
      sinModificar(img_path)


modificacionesAleatorias("directorioAModificar")




