import os
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.geometry import Polygon


def plot_N(images):
    fig = plt.figure(figsize=(10 * len(images), 10))
    gs = GridSpec(nrows=1, ncols=len(images))
    gs.update( hspace = 0.5, wspace = 0.)
    for i in range(len(images)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()  
    
        
def check_collision(tar_poly, others):
    if len(others) == 0:
        return True
    p1 = Polygon(tar_poly)
    for (_, _, poly, _) in others:
        p2 = Polygon(poly)
        if p1.intersects(p2): return False;
    return True
    
                                                        
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)
                                                        
    
def extract_bbox(cx, cy, tar_w, tar_h):
    min_x = cx - tar_w // 2
    max_x = min_x + tar_w
    min_y = cy - tar_h // 2
    max_y = min_y + tar_h
    return np.array([[min_x, min_y], 
                     [min_x, max_y], 
                     [max_x, max_y], 
                     [max_x, min_y],
                     [min_x, min_y]])
    
    
def mod_coords(bbox, delta_w, delta_h):
    new_bbox = bbox.copy()
    new_bbox[:, 0] += delta_w
    new_bbox[:, 1] += delta_h
    return new_bbox


def add_gauss_noise(image, noise_typ='gauss'):
    row, col, ch= image.shape
    mean = 0
    var = np.random.rand(1) * 100
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.zeros(image.shape, np.float32)
    noisy_image[:, :, 0] = image[:, :, 0] + gauss
    noisy_image[:, :, 1] = image[:, :, 1] + gauss
    noisy_image[:, :, 2] = image[:, :, 2] + gauss
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    noisy_image[image == 255] = 255
    return noisy_image
    
    
def crop_w_minimum_bbox(img):
    h, w, _ = img.shape
    left_w, right_w, top_h, bottom_h = 0, w - 1, 0, h - 1
    for i in range(0, w // 2):
        if np.any(img[:, i] < 255):
            left_w = i; break;

    for i in range(w - 1, w // 2, -1):
        if np.any(img[:, i] < 255):
            right_w = i; break;

    for i in range(0, h // 2):
        if np.any(img[i, :] < 255):
            top_h = i; break;
            
    for i in range(h - 1, h // 2, -1):
        if np.any(img[i, :] < 255):
            bottom_h = i; break;

    cropped_img = img[top_h:bottom_h + 1, left_w: right_w + 1]
    return cropped_img, (top_h, left_w)
        