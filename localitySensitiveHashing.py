#https://github.com/mendesk/image-ndd-lsh/blob/master/detect.py
#http://infolab.stanford.edu/~ullman/mining/2006/lectureslides/cs345-lsh.pdf

import argparse
import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image

import datasetModificado.modificaImagen


def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """
    Calculate the dhash signature of a given file

    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2

    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    try:
        pil_image = Image.open(image_file).convert("L").resize(
            (hash_size + 1, hash_size),
            Image.ANTIALIAS)
        dhash = imagehash.dhash(pil_image, hash_size)
        signature = dhash.hash.flatten()
        pil_image.close()
        return signature
    except IOError as e:
        raise e




def lsh(img_path1,img_path2,threshold: float, hash_size: int, bands: int):
    rows: int = int(hash_size ** 2 / bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]


    #calculo la firma de la primera imagen
    signature = calculate_signature(img_path1, hash_size)
    print("calculo la firma de "+img_path1)

    # Keep track of each image's signature
    signatures[img_path1] = np.packbits(signature)

    # Locality Sensitive Hashing
    for i in range(bands):
            signature_band = signature[i * rows:(i + 1) * rows]
            signature_band_bytes = signature_band.tostring()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(img_path1)

    signature2 = calculate_signature(img_path2, hash_size)
    print("calculo la firma de " + img_path2)

    # Keep track of each image's signature
    signatures[img_path2] = np.packbits(signature2)

    # Locality Sensitive Hashing
    for i in range(bands):
        signature_band = signature2[i * rows:(i + 1) * rows]
        signature_band_bytes = signature_band.tostring()
        if signature_band_bytes not in hash_buckets_list[i]:
            hash_buckets_list[i][signature_band_bytes] = list()
        hash_buckets_list[i][signature_band_bytes].append(img_path2)






    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i + 1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i], hash_bucket[j]])
                        )

    # Check candidate pairs for similarity
    near_duplicates = list()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
            np.unpackbits(signatures[cpa]),
            np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size ** 2 - hd) / hash_size ** 2
        if similarity > threshold:
            near_duplicates.append((cpa, cpb, similarity))

    # Sort near-duplicates by descending similarity and return
    near_duplicates.sort(key=lambda x: x[2], reverse=True)
    return near_duplicates



def find_near_duplicates(input_dir: str, threshold: float, hash_size: int, bands: int) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images

    Args:
        input_dir: Directory with images to check
        threshold: Images with a similarity ratio >= threshold will be considered near-duplicates
        hash_size: Hash size to use, signatures will be of length hash_size^2
        bands: The number of bands to use in the locality sensitve hashing process

    Returns:
        A list of near-duplicates found. Near duplicates are encoded as a triple: (filename_A, filename_B, similarity)
    """
    rows: int = int(hash_size ** 2 / bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]

    # Build a list of candidate files in given input_dir
    try:
        file_list = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]
    except OSError as e:
        raise e

    # Iterate through all files in input directory
    for fh in file_list:
        try:
            signature = calculate_signature(fh, hash_size)
        except IOError:
            # Not a PIL image, skip this file
            continue

        # Keep track of each image's signature
        signatures[fh] = np.packbits(signature)

        # Locality Sensitive Hashing
        for i in range(bands):
            signature_band = signature[i * rows:(i + 1) * rows]
            signature_band_bytes = signature_band.tostring()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(fh)

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i + 1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i], hash_bucket[j]])
                        )

    # Check candidate pairs for similarity
    near_duplicates = list()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
            np.unpackbits(signatures[cpa]),
            np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size ** 2 - hd) / hash_size ** 2
        return similarity
    # Sort near-duplicates by descending similarity and return



    # # Argument parser
    # parser = argparse.ArgumentParser(
    #     description="Efficient detection of near-duplicate images using locality sensitive hashing")
    # parser.add_argument("-i", "--inputdir", type=str, default="", help="directory containing images to check")
    # parser.add_argument("-t", "--threshold", type=float, default=0.9, help="similarity threshold")
    # parser.add_argument("-s", "--hash-size", type=int, default=16,
    #                     help="hash size to use, signature length = hash_size^2", dest="hash_size")
    # parser.add_argument("-b", "--bands", type=int, default=16, help="number of bands")

#datasetModificado.modificaImagen.recortar("/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/B.jpg")
#datasetModificado.modificaImagen.recortar("/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/B_MOD1.jpg")
#datasetModificado.modificaImagen.recortar("/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/A.jpg")


input_dir = "/home/claudia/PycharmProjects/similitudImagenes/directorios/IMAGEN1/"
threshold = 0.0
hash_size =8

bands =10

similitud= lsh("directorios/IMAGEN1/IMAGEN1_original.jpg", "directorios/IMAGEN1/IMAGEN1_rotadal.jpg", threshold, hash_size, bands)
if(len(similitud)!=0):
    print(similitud[0][2])
else:
    print(1)
#         if near_duplicates:
#             print(f"Found {len(near_duplicates)} near-duplicate images in {input_dir} (threshold {threshold:.2%})")
#             for a, b, s in near_duplicates:
#                 print(f"{s:.2%} similarity: file 1: {a} - file 2: {b}")
#         else:
#             print(f"No near-duplicates found in {input_dir} (threshold {threshold:.2%})")
# except OSError:
#         print(f"Couldn't open input directory {input_dir}")
#
