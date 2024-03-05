import os
import sys
import shutil
import glob
import random
import json
import cv2
import time
import numpy as np
import argparse
from PIL import Image

from data.data_loader import *
from data.data_utils import build_map_mask, build_point_mask, build_text_mask
from func import sample_symbol_image, create_symbol_images, synthesize

Image.MAX_IMAGE_PIXELS = None 

SYMBOL_BASE_SIZE = {
    "drill_hole": [(30, 30)],
    "gravel_pit": [(30, 30), (60, 60)],
    "inclined_bedding": [(18, 60)],
    "inclined_flow_banding": [(18, 60)],
    "inclined_metamorphic": [(18, 60)],
    "lineation": [(28, 100)],
    "lineation_num": [(18, 60)],
    "mine_shaft": [(30, 30), (60, 60)],
    "overturned_bedding": [(18, 60)],
    "prospect": [(30, 30), (60, 60)],
    "quarry": [(31, 40), (56, 71)],
    "mine_tunnel": [(18, 60)]
}
    

SYMBOL_WITH_NUM = ["inclined_bedding", 
                   "inclined_flow_banding",
                   "inclined_metamorphic",
                   "lineation_num",
                   "overturned_bedding"]

    
def is_valid_mask(mask):
    if np.sum(mask) > crop_size ** 2 / 4:
        return False
    if mask.shape[0] < crop_size or mask.shape[1] < crop_size:
        return False
    return True

    
def main():
    output_json = {'images': [], 'annotations': [], 'img2anno': {}}
    
    image_id = 0
    for map_tif_idx, map_tif in enumerate(candidate_map_tifs):
        
        print(f'>>> Processing candidate map tif: {map_tif_idx} ')
        start_time = time.time()
        
        map_tif_file = os.path.join(map_tif_dir, map_tif)
        map_name = map_tif.split('.')[0]
        map_img = Image.open(map_tif_file)
        map_img = np.array(map_img)        
        
        # load the symbol tif for excluding the patches containing the target symbol
        if map_tif in tar_map_tifs and use_target_symbol_tifs:
            idx = tar_map_tifs.index(map_tif)
            tar_symbol_tif_file = os.path.join(annotation_tif_dir, map_name + '_' + tar_symbol_names[idx] + '.tif')
            tar_symbol_tif = load_symbol_tif(tar_symbol_tif_file)

        # build masks
        map_mask = build_map_mask(map_name, map_mask_dir, map_tif_dir)
        point_mask = build_point_mask(map_name, map_img, annotation_tif_dir) 
        text_mask = build_text_mask(map_name, map_img, spotting_dir)
        
        # The map mask for placing a valid point symbol, 0 means empty, 1 means occupied
        map_mask = map_mask | point_mask | text_mask

        # compute the region of interest
        xs, ys = np.where(map_mask)
        roi_min_x, roi_max_x = np.min(xs), np.max(xs)
        roi_min_y, roi_max_y = np.min(ys), np.max(ys)

        for idx in range(roi_min_y, roi_max_y, shift_size):
            for jdx in range(roi_min_x, roi_max_x, shift_size):
                
                # skip the patch if there are annotations for the target symbol
                if map_tif in tar_map_tifs and use_target_symbol_tifs:
                    tar_symbol_clip = tar_symbol_tif[idx: idx + crop_size, jdx: jdx + crop_size]
                    if np.sum(tar_symbol_clip) > 0: 
                        continue;
                    
                # skip the patch if there are no place to put target symbol
                map_mask_clip = map_mask[idx: idx + crop_size, jdx: jdx + crop_size]
                if not is_valid_mask(map_mask_clip):
                    continue
                    
                map_img_clip = map_img[idx: idx + crop_size, jdx: jdx + crop_size]
                
                # random load target symbol
                symbol_img = sample_symbol_image(target_symbol_images, SYMBOL_BASE_SIZE[target_symbol])
                sample_symbol_images = create_symbol_images(symbol_img, 
                                                            max_rotate=max_rotate, 
                                                            add_num_img=add_num_img,
                                                            number_img_dir=number_img_dir)
                
                output, valid_placements = synthesize(sample_symbol_images, 
                                                      map_img_clip, 
                                                      map_mask_clip, 
                                                      max_count=max_count, 
                                                      allow_collision=allow_collision, 
                                                      require_hollow=require_hollow)
                        
                # output 
                output_image_filename = '{:07d}.jpg'.format(image_id)
                cv2.imwrite(os.path.join(output_image_dir, output_image_filename), output)
                output_json['images'].append({
                    'id': image_id,
                    'image_file': output_image_filename})

                anno_ids = []
                for (x, y, rotated_bbox, degree) in valid_placements:
                    anno_id = len(output_json['annotations'])
                    anno_ids.append(anno_id)
                    output_json['annotations'].append({
                        'id': anno_id,
                        'image_id': image_id,
                        'cx': x,
                        'cy': y,
                        'degree': degree,
                        'poly': rotated_bbox.reshape(-1).tolist()})

                output_json['img2anno'][output_image_filename] = anno_ids
                image_id += 1
        
        if image_id > max_num_synthetic_images:
            break
            
        print('Finsh time: ', time.time() - start_time)
        
    print(f'Finsh generating {image_id} images.')
    with open(output_json_file, 'w') as f:
        json.dump(output_json, f)

    
if __name__ == "__main__":
    
    # basic parameters
    parser = argparse.ArgumentParser(description='symbol')
    parser.add_argument('--root1', type=str, default="./ta1_feature_extraction/")
    parser.add_argument('--root2', type=str, default="./symbol_data/cropped_point_symbols/")    
    parser.add_argument('--target_symbol', type=str, required=True)
    parser.add_argument('--max_rotate', type=int, default=0)
    parser.add_argument('--use_target_symbol_tifs', action='store_false')
    parser.add_argument('--max_num_synthetic_images', type=int, default=10)
    parser.add_argument('--max_count_per_image', type=int, default=10)    
    parser.add_argument('--allow_collision', action='store_true')
    args = parser.parse_args()
    
    # running example
    # python create_synthetic_maps_main.py --target_symbol inclined_flow_banding --max_rotate 45

    root1 = args.root1
    root2 = args.root2
    target_symbol = args.target_symbol
    
    if SYMBOL_BASE_SIZE.get(target_symbol) is None:
        print("Base size for target symbol is missing. Abort.")
        sys.exit(0)
        
    # define parameters
    use_target_symbol_tifs = args.use_target_symbol_tifs
    max_num_synthetic_images = args.max_num_synthetic_images
    max_rotate = args.max_rotate
    max_count = args.max_count_per_image
    allow_collision = args.allow_collision
    add_num_img = True if target_symbol in SYMBOL_WITH_NUM else False
    require_hollow = True if target_symbol in ['drill_hole'] else False
    
    # define path
    map_tif_dir = os.path.join(root1, 'training')
    annotation_tif_dir = os.path.join(root1, 'training_json_gts')
    target_symbol_image_dir = os.path.join(root2, f'draft/{target_symbol}')
    spotting_dir = './mapkurator_result/ta1_competition_train'
    map_mask_dir = os.path.join(root1, 'training_masks')
    number_img_dir = './symbol_data/numbers/'

    output_dir = os.path.join(root1, f'point_synthetic_maps/{target_symbol}')
    output_json_file = os.path.join(output_dir, 'train_poly.json')
    output_image_dir = os.path.join(output_dir, 'train_images')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_image_dir)
    
    # get official symbol images
    target_symbol_images = load_symbol_images(target_symbol_image_dir)
    # target_symbol_shapes = load_symbol_shapes(target_symbol_legend_json)
            
    crop_size = 1024
    shift_size = 512
    
    # extract candidate maps
    all_tif_files = sorted(glob.glob(os.path.join(map_tif_dir, '*.tif')))
    tar_map_tifs, tar_symbol_names = load_symbol_info(target_symbol)
    
    candidate_map_tifs = []    
    if use_target_symbol_tifs:
        candidate_map_tifs += tar_map_tifs
    candidate_map_tifs += find_candidate_maps(tar_map_tifs, all_tif_files, num_candidate_maps=100)
    print(f'Number of candidate map tifs: ', len(candidate_map_tifs), '\n')
    
    main()