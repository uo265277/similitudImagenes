


#se le pasa una lista der sublistas ->[comparador, valor]
#flags
# 0 escalagrises
# 1 normalizado
# 2 clahe
# 3 hog
# 4 gabor
# 5 sift_sim
# 6 ssim
# 7 mse
# 8 gabor_sift_sim

#obtiene el valor máximo y divide todos por él
def obtieneMaxComparadores(resultados, flags):
    cont=0

    #se crea una lista para guardar los maximos resultados de cada comparador, en caso de que no haya
    #un comparador el maximo se marca como 1
    maximos=[]
    longListaMax=flags.count(1)
    rangos=[]
    for i in range(longListaMax):
        maximos.append(0)
        rangos.append("")
    for resultado in  resultados:
        cont = 0
        for sublista in resultado:

            comparador=sublista[0]
            resultado=sublista[1]

            #if flags [0]==1  d
            if(comparador=="escalaGrises"):
                if (resultado> maximos[cont]) :
                    maximos[cont]=resultado
                    rangos[cont]=["escalaGrises",resultado]
                cont=cont+1

            if (comparador == "normalizado"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont] = ["normalizado", resultado]
                cont=cont+1

            if (comparador == "clahe"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["clahe",resultado]
                cont=cont+1


            if (comparador == "hog"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["hog",resultado]
                cont=cont+1


            if (comparador == "gabor"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["gabor",resultado]
                cont=cont+1

            if (comparador == "sift_sim"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["sift_sim",resultado]
                cont=cont+1


            if (comparador == "ssim"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["ssim",resultado]
                cont=cont+1


            if (comparador == "mse"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["mse",resultado]
                cont=cont+1



            if (comparador == "gabor_sift_sim"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["gabor_sift_sim",resultado]
                cont=cont+1

            if (comparador == "psnr"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["psnr",resultado]
                cont=cont+1

            if (comparador == "lsh"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["lsh",resultado]
                cont=cont+1

            if (comparador == "histogramaColor"):
                if (resultado > maximos[cont]):
                    maximos[cont] = resultado
                    rangos[cont]=["histogramaColor",resultado]
                cont=cont+1

    return rangos



#maximos=obtieneMaxComparadores([['escalaGrises', 0.48973268], ['normalizado', 0.30431348], ['clahe', 0.49163023], ['gabor', 0.32423025], ['sift_sim', 0.7329192546583851], ['mse', 26264.01004464286], ['gabor_sift_sim', 0.04195804195804196]],[1,1,1,1,1,1,0,1,1])
#print(maximos)




#funcion que dado un diccionario global devuelve una lista de todos los subdicionarios con profundidad 2
def sacarLista (dir, lista):
    for key, value in dir.items():
        #print(value)
        if isinstance(value, list):
            lista.append(value)
        else:
            sacarLista(value, lista)

    return lista



#def aplicaNormalizacion
#dado un diccionario global aplica a cada sublista [comparador, valor] una modificacion -> [comparador , valor/max]
def aplicaNormalizacion(diccionarioGlobal , maximos):
    for key, value in diccionarioGlobal.items():
        #print(value)
        if isinstance(value, list):
            #LLEGO A LA LISTA DE SUBLISTAS
            #para cada sublista le aplico el valor de maximos
            cont=0
            for sublista in value:
                sublista[1]=sublista[1]/maximos[cont]
                cont=cont+1

        else:
            aplicaNormalizacion(value, maximos)

    return diccionarioGlobal















