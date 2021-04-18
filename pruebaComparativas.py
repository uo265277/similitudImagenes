
from flags import escalaGrises, normalizado, clahe, hog, gabor, sift_sim, ssim, mse


#flags
# 0 escalagrises
# 1 normalizado
# 2 clahe
# 3 hog
# 4 gabor
# 5 sift_sim
# 6 ssim
# 7 mse
# 8 gabor + sift sim

max0=0
max1=0
max2=0
max3=0
max4=0
max5=0
max6=0
max7=0
max8=0



def comparaImagenes(path1, path2, flags):
    listaResultados = []
    #flag escala grises activado
    if (flags[0]==1):
        string0="escalaGrises: "
        result0 = escalaGrises(path1, path2)
        list0=[string0,result0]
        listaResultados.append(list0)
        print(listaResultados)
    #flag normalizado activado
    if (flags[1] == 1):
        string1= "normalizado: "
        result1 = normalizado(path1, path2)
        list1 = [string1, result1]
        listaResultados.append(list1)
    #flag clahe activado
    if (flags[2] == 1):
        string2 = "clahe: "
        result2 = clahe(path1, path2)
        list2 = [string2, result2]
        listaResultados.append(list2)
    #flag hog activado
    # if (flags[3] == 1):
    #     string3 = "hog: "
    #     result3 = hog(path1, path2)
    #     if (result3 > max3):
    #         result3 = max3
    #     list3 = [string3, result3]
    #     listaResultados.__add__(list3)
    #flag gabor activado
    if (flags[4] == 1):
        string4 = "gabor: "
        result4 = gabor(path1, path2)
        list4 = [string4, result4]
        listaResultados.append(list4)
    #flag sift_sim activado
        if (flags[5] == 1):
            string5 = "sift_sim: "
            result5 = sift_sim(path1, path2)
            list5 = [string5, result5]
            listaResultados.append(list5)
    #flag sift_sim activado
            if (flags[6] == 1):
                string6 = "ssim: "
                result6 = ssim(path1, path2)
                list6 = [string6, result6]
                listaResultados.append(list6)
    #flag sme activado
            if (flags[7] == 1):
                string7= "mse: "
                result7 = mse(path1, path2)
                list7 = [string7, result7]
                listaResultados.append(list7)

    return listaResultados

resultados=comparaImagenes(r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\CLAUDIA\Claudia-004.jpg",
                r"C:\Users\claud\Desktop\CUARTO_ING_INF\TFG\similitudImagenes\datasetModificado\FRANK\frank-0004.jpg",
                [1,1,1,1,1,1,1,1])

print(resultados)