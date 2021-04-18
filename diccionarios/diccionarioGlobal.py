import ayudaDirectorios


#diccionario que devuelve asociado a el path de la imagen una lista de pares comparador-resultado
#siend los pares : comparativa normal escala grises -> doistancia euclídea



#metodo normalizar luista que recibe un diccionario con pares string valor y normaliza el valor entre 0 y 1

















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
        dirK = ayudaDirectorios.directorioImagen(k)

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
        nomK = ayudaDirectorios.nombreImagen(k)
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

            dirK2 = ayudaDirectorios.directorioImagen(k2)
            nomK2 = ayudaDirectorios.nombreImagen(k2)
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
