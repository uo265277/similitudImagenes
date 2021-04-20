


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
    for flag in flags:
        if (flag==1):
            maximos.append(0)
        else:
            maximos.append(1)

    for sublista in  resultados:
        comparador=sublista[0]
        resultado=sublista[1]
        if(comparador=="escalaGrises"):
            if (resultado> maximos[0]) : maximos[0]=resultado
        if (comparador == "normalizado"):
            if (resultado > maximos[1]): maximos[1] = resultado

        if (comparador == "clahe"):
            if (resultado > maximos[2]): maximos[2] = resultado

        if (comparador == "hog"):
            if (resultado > maximos[3]): maximos[3] = resultado

        if (comparador == "gabor"):
            if (resultado > maximos[4]): maximos[4] = resultado
        if (comparador == "sift_sim"):
            if (resultado > maximos[5]): maximos[5] = resultado
        if (comparador == "ssim"):
            if (resultado > maximos[6]): maximos[6] = resultado
        if (comparador == "mse"):
            if (resultado > maximos[7]): maximos[7] = resultado
        if (comparador == "gabor_sift_sim"):
            if (resultado > maximos[8]): maximos[8] = resultado
    return maximos



maximos=obtieneMaxComparadores([['escalaGrises', 0.48973268], ['normalizado', 0.30431348], ['clahe', 0.49163023], ['gabor', 0.32423025], ['sift_sim', 0.7329192546583851], ['mse', 26264.01004464286], ['gabor_sift_sim', 0.04195804195804196]],[1,1,1,1,1,1,0,1,1])
print(maximos)




#funcion que dado un diccionario global devuelve una lista de todos los subdicionarios con profundidad 2
def obtieneResultadosGlobales(diccionarioGlobal):

