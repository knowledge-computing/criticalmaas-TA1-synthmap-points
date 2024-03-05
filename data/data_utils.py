import os
import json
import glob
import numpy as np
import cv2
from PIL import Image

from data.cropping_worker import cropping_worker
    
        
def build_map_mask(map_name, mask_root, data_root):
    mask_file = os.path.join(mask_root, 'crop_' + map_name + '.tif')
    if os.path.exists(mask_file):
        map_mask = Image.open(mask_file)
        map_mask = np.array(map_mask) == 0
    else:
        print('Cannot find an existing mask: ', mask_root, '\n')
        map_mask = cropping_worker(None, map_name + '.json', data_root, './tmp', True)
    return map_mask

    
def build_point_mask(map_name, map_img, data_root):
    with open ('data/map_list.json', 'r') as f:
        map_list = json.load(f)

    pt_tifs = []
    for tif_file in map_list[map_name + '.tif']:
        pt_tifs.append(os.path.join(data_root, tif_file))
    print(f'Number of point features in `{map_name}`: {len(pt_tifs)}')
    
    point_mask = np.zeros((map_img.shape[0], map_img.shape[1]))
    try:
        for tif in pt_tifs:
            im = Image.open(tif)
            point_mask += np.array(im)

        kernel = np.ones((5, 5), np.uint8) 
        point_mask = cv2.dilate(point_mask, kernel, iterations=5)
        return point_mask > 0
    
    except Exception as e:
        return point_mask > 0
    
    
def build_text_mask(map_name, map_img, spotting_root):
    spotting_output_file = os.path.join(spotting_root, map_name + '.geojson')
    text_mask = np.zeros((map_img.shape[0], map_img.shape[1]))
    if not os.path.exists(spotting_output_file):
        return text_mask > 0
    try:
        with open(spotting_output_file, 'r') as f:
            spotting_output = json.load(f)

        for feature in spotting_output['features']:
            poly = np.abs(np.array(feature['geometry']['coordinates'])).reshape(-1, 2).astype(int)
            cv2.drawContours(text_mask, [poly], 0, 1, -1)

        # kernel = np.ones((5, 5), np.uint8) 
        # text_mask = cv2.dilate(text_mask, kernel, iterations=3)
        kernel = np.ones((3, 3), np.uint8) 
        text_mask = cv2.erode(text_mask, kernel, iterations=5)
        return text_mask > 0
    
    except Exception as e:
        return text_mask > 0
    
    