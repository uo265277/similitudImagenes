import ayudaDirectorios
from calculoRangos import estaEnRango, rangosImagenesIndividuales
from pruebaComparativas import comparaImagenes


def calculaDiccionario(imagenes, flags, maximos, opcion):
    # entrada -> lista de sublistas -> [PATH_IMAGEN, DIRECTORIO_PADRE]

    # CREO EL DICCIONARIO GLOBAL
    diccionarioGlobal: dict = {}
    cont =0
    dirAux =""
    with open("resultados.txt","w") as f:
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
            cambio=False
            plagios=0
            directorioCambiado=True
            anteriorDirectorio=""
            for k2 in imagenes:

                #si el anterior directorio no es como dirK2 saco los resultados de anteriordirectorio y plagios y pongo
                #plagios a 0 a no ser que anterior directorio sea ""

                dirK2 = ayudaDirectorios.directorioImagen(k2)
                nomK2 = ayudaDirectorios.nombreImagen(k2)
                if(anteriorDirectorio!=dirK2 and anteriorDirectorio!=""):

                    f.write("Se ha contabilizado un total de "+str(plagios)+" posibles plagios entre el trabajo"+str(dirK)+" y el trabajo "+str(anteriorDirectorio)+'\n')
                    plagios=0
                    anteriorDirectorio=dirK2


                if(dirK !=dirK2):
                   listaResultados=comparaImagenes(k, k2, flags)

                   if(opcion=="a" or opcion=="c"):
                       #miro si alguno de esos resultados considera como plagio la imagen = resultado < maximo = resultado
                       # pertenece a rango
                       pertenece=estaEnRango(listaResultados,maximos)
                       #si pertenece lo añado a resultados
                       if(pertenece):
                           diccionarioImagen[dirK2 + nomK2]=listaResultados
                           plagios=plagios+1
                           f.write("Posible plagio detectado entre la imagen "+str(k) +" del directorio "+str(dirK) +" y la imagen "+str(k2)+" del directorio "+ str(dirK2)+'\n')

                   else:
                       #en caso de estar en la opcion b, se individualiza los maximos por imagen
                       maximos = rangosImagenesIndividuales(flags, k)
                       pertenece = estaEnRango(listaResultados, maximos)
                       # si pertenece lo añado a resultados
                       if (pertenece):
                           diccionarioImagen[dirK2 + nomK2] = listaResultados
                           plagios = plagios + 1
                           f.write("Posible plagio detectado entre la imagen "+k +" del directorio "+dirK +" y la imagen "+k2+" del directorio "+ dirK2+'\n')
                #cambiamos de directorio

            # posible filtro aqui mas tarde
            #  print("diff es")
            #  print(diff)
            #  distancias[k]=diff
    return diccionarioGlobal




