import os
import json
import glob
import cv2
import numpy as np
import random
from PIL import Image



def load_symbol_info(symbol):
    with open ('data/pnt_symbol_map_list.json', 'r') as f:
        pnt_symbol_map_list = json.load(f)
    map_tifs = [k for k, _ in pnt_symbol_map_list[symbol].items()]
    symbol_names = [v[0] for _, v in pnt_symbol_map_list[symbol].items()]
    print(f'Maps contain `{symbol}`:', map_tifs)
    print(f'Corresponding point symbol names:', symbol_names, '\n')
    return map_tifs, symbol_names


def load_symbol_tif(symbol_tif_file):
    symbol_tif = Image.open(symbol_tif_file)
    symbol_tif = np.array(symbol_tif)
    kernel = np.ones((5, 5), np.uint8) 
    symbol_tif = cv2.dilate(symbol_tif, kernel, iterations=5)
    return symbol_tif
    
    
def load_symbol_images(symbol_image_dir):
    return glob.glob(os.path.join(symbol_image_dir, '*.png'))


def load_symbol_shapes(symbol_legend_json):
    with open(symbol_legend_json, 'r') as f:
        symbol_legends = json.load(f)
    
    symbol_shapes, _sum = [], []
    for legend in symbol_legends:
        shape = symbol_legends[legend]
        symbol_shapes.append(shape)
        _sum.append(shape[0] + shape[1])
                
    if len(symbol_shapes) < 5:
        return symbol_shapes
    else:
        _sum.sort() 
        mid = len(_sum) // 4
        output_shapes = []
        for shape in symbol_shapes:
            if shape[0] + shape[1] < _sum[mid]:
                output_shapes.append(shape)
        return output_shapes
        
    
# def find_candidate_maps(target_symbol):
#     with open ('data/pnt_symbol_map_list_new.json', 'r') as f:
#         pnt_symbol_map_list = json.load(f)
#     target_map_tifs = list(pnt_symbol_map_list[target_symbol].keys())
    
#     correlated_symbols = []
#     for symbol, li in pnt_symbol_map_list.items():
#         map_tifs = list(li.keys())
#         for tif in map_tifs:
#             if tif in target_map_tifs:
#                 correlated_symbols.append(symbol)
#     correlated_symbols = list(set(correlated_symbols))
#     print(f'Number of correlated symbols with {target_symbol} = {len(correlated_symbols)} \n')

#     candidate_map_tifs = []
#     for symbol, li in pnt_symbol_map_list.items():
#         if symbol not in correlated_symbols: continue;
#         map_tifs, _ = li
#         for tif in map_tifs:
#             if tif not in target_map_tifs:
#                 candidate_map_tifs.append(tif)
#     print(f'Number of candidate maps = {len(candidate_map_tifs)} \n')
    
#     if len(candidate_map_tifs) < 100:
#         all_tifs = []
#         for symbol, li in pnt_symbol_map_list.items():
#             all_tifs += li[0]
#         print(f'Randomly select another 100 candidate maps from {len(all_tifs)} maps \n')
#         candidate_map_tifs += random.sample(all_tifs, 100)
        
#     return list(set(candidate_map_tifs))


def find_candidate_maps(tar_map_tifs, all_tifs, num_candidate_maps=100):
    candidate_map_tifs = []
    for tif in all_tifs:
        if tif not in tar_map_tifs:
            candidate_map_tifs.append(tif)
    candidate_map_tifs = random.sample(candidate_map_tifs, num_candidate_maps)
    return candidate_map_tifs
