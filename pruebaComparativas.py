
from flags import escalaGrises, normalizado, clahe, hog, gabor, sift_sim, ssim, mse, gabor_sift_sim, psnr, lsh, \
    histogramaColor


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
#9 psnr
#10 lsh
#11 histograma color


def comparaImagenes(path1, path2, flags):
    listaResultados = []
    #flag escala grises activado
    if (flags[0]==1):
        string0="escalaGrises"
        result0 = escalaGrises(path1, path2)
        list0=[string0,result0]
        listaResultados.append(list0)
    #flag normalizado activado
    if (flags[1] == 1):
        string1= "normalizado"
        result1 = normalizado(path1, path2)
        list1 = [string1, result1]
        listaResultados.append(list1)
    #flag clahe activado
    if (flags[2] == 1):
        string2 = "clahe"
        result2 = clahe(path1, path2)
        list2 = [string2, result2]
        listaResultados.append(list2)
    #flag hog activado
    if (flags[3] == 1):
        string3 = "hog"
        result3 = hog(path1, path2)
        list3 = [string3, result3]
        listaResultados.append(list3)
    #flag gabor activado
    if (flags[4] == 1):
        string4 = "gabor"
        result4 = gabor(path1, path2)
        list4 = [string4, result4]
        listaResultados.append(list4)
    #flag sift_sim activado
        if (flags[5] == 1):
            string5 = "sift_sim"
            result5 = sift_sim(path1, path2)
            list5 = [string5, result5]
            listaResultados.append(list5)
    #flag sift_sim activado
            if (flags[6] == 1):
                string6 = "ssim"
                result6 = ssim(path1, path2)
                list6 = [string6, result6]
                listaResultados.append(list6)
    #flag sme activado
            if (flags[7] == 1):
                string7= "mse"
                result7 = mse(path1, path2)
                list7 = [string7, result7]
                listaResultados.append(list7)
    #flag GABOR+SIFT_SIM activado
            if (flags[8] == 1):
                string8= "gabor_sift_sim"
                result8 = gabor_sift_sim(path1, path2)
                list8 = [string8, result8]
                listaResultados.append(list8)
    #flag psnr activado
            if (flags[9] == 1):
                string9 = "psnr"
                result9 = psnr(path1, path2)
                list9 = [string9, result9]
                listaResultados.append(list9)
    #flag lsh activado
            if (flags[10] == 1):
                string10 = "lsh"
                result10 = lsh(path1, path2)
                list10 = [string10, result10]
                listaResultados.append(list10)
    #flag histograma color activado
            if (flags[11] == 1):
                string11 = "histogramaColor"
                result11 = histogramaColor(path1, path2)
                list11 = [string11, result11]
                listaResultados.append(list11)

    return listaResultados


listaResultados=comparaImagenes("directorios/IMAGEN4/IMAGEN4_original.jpg", "directorios/IMAGEN4/IMAGEN4_parecida.jpg", [1,1,1,1,1,1,1,1,1,1,1,1])
print(listaResultados)