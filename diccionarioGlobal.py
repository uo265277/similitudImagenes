import ayudaDirectorios
from calculoRangos import estaEnRango, rangosImagenesIndividuales
from pruebaComparativas import comparaImagenes


def calculaDiccionario(imagenes, flags, maximos, opcion):
    # entrada -> lista de sublistas -> [PATH_IMAGEN, DIRECTORIO_PADRE]

    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""

    for k in imagenes:


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
        # print("Estado del diccionadio de pruebas_directorio "+ dirK+" : ")
        # print(diccionarioDirectorios)
        # print("fin")
        for k2 in imagenes:

            dirK2 = ayudaDirectorios.directorioImagen(k2)
            nomK2 = ayudaDirectorios.nombreImagen(k2)
            if(dirK !=dirK2):
               listaResultados=comparaImagenes(k, k2, flags)

               if(opcion=="a" or opcion=="c"):
                   #miro si alguno de esos resultados considera como plagio la imagen = resultado < maximo = resultado
                   # pertenece a rango
                   pertenece=estaEnRango(listaResultados,maximos)
                   #si pertenece lo añado a resultados
                   if(pertenece): diccionarioImagen[dirK2 + nomK2]=listaResultados

               else:
                   #en caso de estar en la opcion b, se individualiza los maximos por imagen
                   maximos = rangosImagenesIndividuales(flags, k)
                   pertenece = estaEnRango(listaResultados, maximos)
                   # si pertenece lo añado a resultados
                   if (pertenece): diccionarioImagen[dirK2 + nomK2] = listaResultados






        # posible filtro aqui mas tarde
        #  print("diff es")
        #  print(diff)
        #  distancias[k]=diff
    return diccionarioGlobal




