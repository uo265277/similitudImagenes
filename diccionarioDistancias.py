
#*******************************************************************************

#devuelve 1 de las 8 diferentes normas de matriz o vector
#¿aqui podria usar la biblioteca annoy para calcular el vecino mas cercano? o mas bien en otra funcion buscando entre todos los v. carac ya que aqui solo
#se comparan el v.caract de img a y de img b
#¿que norma es manhattan o la distancia euclidea? ¿cual es mejor?
def findDifference(f1, f2):
    return np.linalg.norm(f1-f2)
    #return manhattan_distance(f1, f2)ç

def distanciaCoseno(f1,f2):
  a1 = np.squeeze(np.asarray(f1))
  a2 = np.squeeze(np.asarray(f2))

  return ( round(np.dot(a1, a2) / (np.linalg.norm(f1) * np.linalg.norm(f2))  ,4) )

def manhattan_distance(a, b):
    return np.abs(a - b).sum()


#NO VA, ALGO DE DIMENSIONES
def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.
        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.
        Args:
            input1, input2: two input numpy arrays.
        Returns:
            Element-wise cosine distances of two inputs.
        '''
        return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
        #return np.dot(input1, input2.T) / \
         #       np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
          #              np.linalg.norm(input2.T, axis=0, keepdims=True))



#*******************************************************************************

# *******************************************************************************





def calculaDiccionarioDistancia(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    distancias: dict = {}
    for k in feature_vectors:
        print("Se procede a calcular las distancias de las demás imagenes con la imagen: " ,k)
        nombreDiccionario ="distacias " +k
        print(nombreDiccionario)
        nombreDiccionario: dict = {}
        distancias[k ]= nombreDiccionario
        for k2 in feature_vectors:


            if(k !=k2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                nombreDiccionario[k2 ] =diff

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return distancias


# *******************************************************************************


# igual que el anterior pero solo se comparan si estan en directorios diferentes
def calculaDiccionarioDistancia2(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    distancias: dict = {}
    for k in feature_vectors:
        dirK =directorioImagen(k)
        print("Se procede a calcular las distancias de las demás imagenes con la imagen: " ,k)
        nombreDiccionario ="distacias " +k
        print(nombreDiccionario)
        nombreDiccionario: dict = {}
        distancias[k ]= nombreDiccionario
        for k2 in feature_vectors:

            dirK2 =directorioImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                nombreDiccionario[k2 ] =diff

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return distancias


# *******************************************************************************


# igual que el anterior pero ordenando por diccionarios
def calculaDiccionarioDistancia3(feature_vectors):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in feature_vectors:


        # CREO EL DICCIONARIO DE DIRECTORIO
        dirK =directorioImagen(k)

        # si estoy en el mismo directorio
        if cont==0:
            print("traza  if cont==0: ")
            cont =cont +1
            dirAux =dirK
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        print()
        print()
        print()
        print()
        print()
        print("*************DICCIONARIO DE DIRECTORIO  " +dirK +" ******************")
        # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

        # si el contador es 0 quiere decir que he cambiado de directorio
        if (dirAux!=dirK):
            print("traza if (dirAux!=dirK): ")
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}
            dirAux =dirK

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        # CREO EL SUBDICCIONARIO DE LA IMAGEN
        nomK =nombreImagen(k)
        diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
        print()
        print("--------------  " +diccionarioImagen +" -----------------------------")
        diccionarioImagen: dict = {}

        # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
        diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
        # print("Estado del diccionadio de directorios "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in feature_vectors:

            dirK2 =directorioImagen(k2)
            nomK2 =nombreImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                # diff=distanciaCoseno(feature_vectors[k],feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)
                diccionarioImagen[dirK2 +nomK2 ] =diff
                # print("hola2")
                # print(diccionarioImagen)
                # print("Estado del diccionadio de directorios "+ dirK+" : ")
                # print(diccionarioDirectorios)
                # print("fin")

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal


# *******************************************************************************


# igual que el anterior pero filtrand con el ratio

def calculaDiccionarioDistancia4(feature_vectors, ratio):
    # entrada -> clave valor imagen y su vector
    # salida -> k, v siendo k imagen comparada con la imagen a;  y v la distancia euclidea

    # diccionario que contiene diccionarios de la imagen con las demas imagenes y su similitud
    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in feature_vectors:


        # CREO EL DICCIONARIO DE DIRECTORIO
        dirK =directorioImagen(k)

        # si estoy en el mismo directorio
        if cont==0:
            print("traza  if cont==0: ")
            cont =con t +1
            dirAux =dirK
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        print()
        print()
        print()
        print()
        print()
        print("*************DICCIONARIO DE DIRECTORIO  " +dir K +" ******************")
        # print("Se procede a calcular las distancias de las demás imagenes con la imagen: ",k)

        # si el contador es 0 quiere decir que he cambiado de directorio
        if (dirAux!=dirK):
            print("traza if (dirAux!=dirK): ")
            diccionarioDirectorios ="DiccionarioDelDirectorio " +dirK
            print(diccionarioDirectorios)
            diccionarioDirectorios: dict = {}
            dirAux =dirK

            # ASIGNO A LA CLAVE DIR* EL DICCIONARIO DE DIR*
            diccionarioGlobal[dirK ]= diccionarioDirectorios

        # CREO EL SUBDICCIONARIO DE LA IMAGEN
        nomK =nombreImagen(k)
        diccionarioImagen ="SubDiccionarioImagen " +dirK +nomK
        print()
        print("--------------  " +diccionarioImagen +" -----------------------------")
        diccionarioImagen: dict = {}

        # ASIGNO A LA CLAVE NOMK EL DICCIONARIO DE NOMK
        diccionarioDirectorios[dirK +nomK ]= diccionarioImagen
        # print("Estado del diccionadio de directorios "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in feature_vectors:

            dirK2 =directorioImagen(k2)
            nomK2 =nombreImagen(k2)
            if(dirK !=dirK2):
                diff =findDifference(feature_vectors[k] ,feature_vectors[k2])
                print("La distancia de ", k, " con " ,k2, "es", diff)

                if(diff<=ratio):
                    diccionarioImagen[dirK2 +nomK2 ] =diff
                # print("hola2")
                # print(diccionarioImagen)
                # print("Estado del diccionadio de directorios "+ dirK+" : ")
                # print(diccionarioDirectorios)
                # print("fin")

        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal


