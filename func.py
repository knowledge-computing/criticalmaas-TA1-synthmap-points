import os
import glob
import random
import json
import math
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Polygon

from utils.utils import * 

Image.MAX_IMAGE_PIXELS = None 
MAX_NUM_SYMBOL_IMAGES = 30
MAX_NUM_ITERS = 10
NUMBER_THRE = 5


def random_resize(img, target_h, target_w):
    """ random resize the symbol images """
    ratio = np.random.rand(1) * 0.25 + 0.8  # 0.8~1.1
    new_h = int(target_h * ratio)
    new_w = int(target_w * ratio)    
    delta = random.randint(-3, 1)
    new_w += delta
    new_h += delta
    img = cv2.resize(img, (new_w, new_h))
    return img

def enlarge_image(img):
    """ enlarge symbol image for rotation """
    h, w, _ = img.shape
    pad_h = int((h * 1.414 + w / 1.414 + 20 - h) // 2 + 50)
    pad_w = int((h * 1.414 + w / 1.414 + 20 - w) // 2 + 50)
    enlarge_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, 
                                     cv2.BORDER_CONSTANT, None, value=(255,255,255)) 
    return enlarge_img, (pad_h, pad_w)
    

def sample_symbol_image(candidate_images, candidate_shapes):
    if len(candidate_images) > 1:
        symbol_file = random.choice(candidate_images)
        symbol_h, symbol_w = random.choice(candidate_shapes)    
    else:
        symbol_file = candidate_images[0]
        symbol_h, symbol_w = candidate_shapes[0]
        
    symbol_img = cv2.imread(symbol_file)
    symbol_img = random_resize(symbol_img, symbol_h, symbol_w)
    
    if random.random() > 0.5:
        kernel = np.ones((2, 2), np.uint8) 
        symbol_img = cv2.erode(symbol_img, kernel, iterations=1)
    # if random.random() > 0.6:
    #     symbol_img = add_gauss_noise(symbol_img)
    
    # print('The selected symbol image:', symbol_file)
    # print(f'The symbol image is resized to: [{symbol_img.shape[0]}, {symbol_img.shape[1]}]')
    return symbol_img
    
    
def sample_number_image(tar_h, number_img_dir):
    num_img = random.choice(glob.glob(os.path.join(number_img_dir, '*.png')))
    num_img = cv2.imread(num_img)
    ratio = tar_h / num_img.shape[0]
    num_img = cv2.resize(num_img, (int(num_img.shape[1]*ratio), int(num_img.shape[0]*ratio)))
    return num_img
    
    
def generate_image_with_number(num_bbox, num_img, symbol_rotated_bbox, rotated_img):
    out_img = None
    if Polygon(num_bbox).intersects(Polygon(symbol_rotated_bbox)):
        tmp_rotated_img = rotated_img.copy()
        min_x, min_y = num_bbox[0]
        max_x, max_y = num_bbox[2]
        tmp_rotated_img[min_y: max_y, min_x: max_x] = 255
        if np.sum(rotated_img==0) - np.sum(tmp_rotated_img==0) < NUMBER_THRE:     
            out_img = np.full_like(rotated_img, 255)
            out_img[min_y: max_y, min_x: max_x] = num_img
            out_img = np.min(np.stack([out_img, rotated_img]), axis=0)            
    return out_img


def gen_hollow_mask(img, output_mask):
    h, w = output_mask.shape[:2]
    bi_tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th, tmp_th = cv2.threshold(bi_tmp, 96, 255, cv2.THRESH_BINARY_INV)
    tmp_floodfill = tmp_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(tmp_floodfill, mask, (0,0), 255);
    tmp_floodfill_inv = cv2.bitwise_not(tmp_floodfill)
    output_mask = np.max(np.stack([output_mask, tmp_floodfill_inv]), axis=0)
    return output_mask
    

def create_symbol_images(symbol_img, max_rotate=0, add_num_img=False, number_img_dir=None):
    out_images = []
    
    symbol_h, symbol_w, _ = symbol_img.shape
    cx, cy = symbol_w // 2, symbol_h // 2
    bbox = extract_bbox(cx, cy, symbol_w, symbol_h)
        
    # enlarge symbol image for rotation
    enlarge_img, (pad_h, pad_w) = enlarge_image(symbol_img) 
    enlarge_h, enlarge_w, _ = enlarge_img.shape  
    enlarge_cx, enlarge_cy = enlarge_w // 2, enlarge_h // 2

    for _ in range(MAX_NUM_SYMBOL_IMAGES):
        if max_rotate > 0:
            degree = random.randint(-max_rotate, max_rotate)
            rotated_bbox = np.array([rotate((cx, cy), pt,  math.radians(-degree)) for pt in bbox])
        else:
            degree = 0  
            rotated_bbox = bbox
        
        M = cv2.getRotationMatrix2D((enlarge_cx, enlarge_cy), degree, 1.0)
        rotated_img = cv2.warpAffine(enlarge_img, M, (enlarge_h, enlarge_w), borderValue=(255,255,255))
        enlarge_rotated_bbox = mod_coords(rotated_bbox, pad_w, pad_h)
        
        out_img = None    
        if add_num_img:
            num_img = sample_number_image(symbol_h, number_img_dir)
            
            # sample number location
            for _ in range(MAX_NUM_ITERS):
                num_cx = enlarge_cx + random.randint(-symbol_w, symbol_w)
                num_cy = enlarge_cy + random.randint(-symbol_h, symbol_h)
                num_bbox = extract_bbox(num_cx, num_cy, num_img.shape[1], num_img.shape[0])
                out_img = generate_image_with_number(num_bbox, num_img, 
                                                     enlarge_rotated_bbox, rotated_img)
                if out_img is not None: 
                    break
        else:
            out_img = rotated_img.copy()
        
        if out_img is None: continue;
        
        out_img, (top_h, left_w) = crop_w_minimum_bbox(out_img)
        out_img = add_gauss_noise(out_img)
        enlarge_rotated_bbox = mod_coords(enlarge_rotated_bbox, -left_w, -top_h)
        out_images.append([out_img, enlarge_rotated_bbox, math.radians(-degree)])
    return out_images
    
        
def synthesize(symbol_images, background, bg_mask, max_count=20, allow_collision=False, require_hollow=False):
    
    bg_h, bg_w, _ = background.shape
    bg_mask = bg_mask.astype(float)

    output = np.full_like(background, [255, 255, 255], dtype=np.uint8)
    output_mask = np.zeros((bg_h, bg_w), dtype=np.uint8) 
    valid_placements = []
    tar_count = min(len(symbol_images), random.randint(1, max_count))
    for (symbol_img, rotated_bbox, degree) in symbol_images:
        
        symbol_h, symbol_w, _ = symbol_img.shape
        # random initialize center point (cx, cy)
        left_bound = symbol_w + abs(min(0, np.min(rotated_bbox[:, 0])))
        right_bound = bg_w - max(symbol_w, np.max(rotated_bbox[:, 0])) - 1
        top_bound = symbol_h + abs(min(0, np.min(rotated_bbox[:, 1])))
        bottom_bound = bg_h - max(symbol_h, np.max(rotated_bbox[:, 1])) - 1
        
        for _ in range(20):
            cx = random.randint(left_bound, right_bound)
            cy = random.randint(top_bound, bottom_bound)
            
            left, top = cx-symbol_w//2, cy-symbol_h//2
            symbol_bbox = mod_coords(rotated_bbox, left, top)

            tmp_symbol_img = np.zeros((bg_h, bg_w))
            tmp_symbol_img = cv2.drawContours(tmp_symbol_img, [symbol_bbox], 0, 1, -1)
            tmp_mask_img = bg_mask.copy()
            tmp_mask_img = cv2.drawContours(tmp_mask_img, [symbol_bbox], 0, 1, -1)
            if np.sum(tmp_mask_img - bg_mask) == np.sum(tmp_symbol_img):
                if allow_collision or check_collision(symbol_bbox, valid_placements):

                    tmp = np.full_like(background, [255, 255, 255], dtype=np.uint8)
                    tmp[top: top+symbol_h, left: left+symbol_w] = symbol_img
                    if require_hollow:
                        output_mask = gen_hollow_mask(tmp, output_mask)

                    output = np.min(np.stack([tmp, output]), axis=0)
                    valid_placements.append([cx, cy, symbol_bbox, degree])
                    break
                    
        if len(valid_placements) == tar_count:
            break
                    
    output = np.min(np.stack([output, background]), axis=0)
    output[output_mask > 0, :] = 255
    return output, valid_placements
                
                
        
# def sample_symbol_locations(mask_img, symbol_img, 
#                             max_count, max_rotate=0, allow_collision=False):
    
#     mask_img = mask_img.astype(float)
#     img_h, img_w = mask_img.shape
#     symbol_h, symbol_w, _ = symbol_img.shape
#     valid_placement = []
#     count, tar_count = 0, random.randint(1, max_count)
#     for _ in range(20):
#         if count == tar_count: break;
            
#         tmp_img = mask_img.copy()
#         tmp_symbol_img = np.zeros((img_h, img_w))
        
#         # random initialize center point (cx, cy)
#         cx = random.randint(symbol_w + 1, img_w - symbol_w - 1)
#         cy = random.randint(symbol_h + 1, img_h - symbol_h - 1)
#         bbox = extract_bbox(cx, cy, symbol_w, symbol_h)
        
#         if max_rotate > 0:
#             degree = random.randint(-max_rotate, max_rotate)
#             rotated_bbox = [rotate((cx, cy), pt,  math.radians(-degree)) for pt in bbox]
#         else:
#             degree = 0    
#             rotated_bbox = bbox

#         rotated_bbox = np.array(rotated_bbox)                                                     
#         tmp_symbol_img = cv2.drawContours(tmp_symbol_img, [rotated_bbox], 0, 1, -1)
#         tmp_img = cv2.drawContours(tmp_img, [rotated_bbox], 0, 1, -1)
#         if np.sum(tmp_img - mask_img) == np.sum(tmp_symbol_img):
#             if allow_collision or check_collision(rotated_bbox, valid_placement):
#                 valid_placement.append((cx, cy, rotated_bbox, degree))
#             count += 1
    
#     if len(valid_placement) < tar_count: return None;
#     else: return valid_placement;
        
        
    
# def merge_two(symbol_locations, symbol_img, background):
    
#     output = np.full_like(background, [255, 255, 255], dtype=np.uint8)
#     h, w, _ = background.shape
#     symbol_h, symbol_w, _ = symbol_img.shape
    
#     for (x, y, rotated_bbox, degree) in symbol_locations:
#         tmp_symbol_img = symbol_img.copy()
#         tmp = np.full_like(background, [255, 255, 255], dtype=np.uint8)
            
#         bbox = extract_bbox(x, y, symbol_w, symbol_h)
#         min_x, min_y = bbox[0]
#         max_x, max_y = bbox[2]
#         tmp[min_y: max_y, min_x: max_x] = tmp_symbol_img

#         M = cv2.getRotationMatrix2D((x, y), degree, 1.0)
#         tmp = cv2.warpAffine(tmp, M, (h, w), borderValue=(255,255,255))
#         output = np.min(np.stack([tmp, output]), axis=0)        

#     output = np.min(np.stack([output, background]), axis=0)
#     return output

