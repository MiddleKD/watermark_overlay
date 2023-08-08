import multiprocessing
from tqdm.contrib.concurrent import process_map
import cv2
import numpy as np
import os
from PIL import Image

def resize_watermark(img, watermark):
    w_height, w_width = watermark.shape[:2]
    ratio = w_width / w_height

    new_height = img.shape[0] // 7
    new_width = int(new_height * ratio)

    resized_watermark = cv2.resize(watermark, (new_width, new_height))

    return resized_watermark

def change_gray(watermark):
    gray_watermark = np.where(0<watermark, 127, watermark)
    return gray_watermark

def random_alpha_modify(tiled_pattern):
    height, width, _ = tiled_pattern.shape
    x = np.random.randint(0, int(width*0.7))
    y = np.random.randint(0, int(height*0.7))
    h = height//2
    w = width//2

    tiled_pattern[y:y+h, x:x+w, 3] *= 0.5

    return tiled_pattern

def visualize_numpy(np_array):
    img = Image.fromarray(np_array.astype(np.uint8))
    return img

def overlay(image, watermark, alpha_range=(0.08, 0.2), alpha_crop=2, gray=True):
    if gray==True:
        watermark = change_gray(watermark)
    
    watermark = resize_watermark(image, watermark)

    tile_x = image.shape[1] // watermark.shape[1] + 1
    tile_y = image.shape[0] // watermark.shape[0] + 1

    tiled_pattern = np.tile(watermark, (tile_y, tile_x, 1))
    tiled_pattern = tiled_pattern[:image.shape[0], :image.shape[1]]

    if alpha_crop:
        for _ in range(alpha_crop):
            tiled_pattern = random_alpha_modify(tiled_pattern)

    pattern_bgr = tiled_pattern[..., :3]
    pattern_alpha = tiled_pattern[..., 3]

    alpha_mask = cv2.merge([pattern_alpha, pattern_alpha, pattern_alpha]) / 255.0 * np.random.uniform(*alpha_range)

    overlay = pattern_bgr * alpha_mask + image * (1 - alpha_mask)
    overlay = overlay.astype(np.uint8)

    return overlay


def save_img(output, save_dir, img_fn, watermark_fn, idx):
    output = output.astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{img_fn}_{watermark_fn}_{idx}.png"), output)


def process(image_tuple, watermark_tuple, save_dir, overlay_iter):
     image_path, img_fn = image_tuple
     image = cv2.imread(image_path).astype(np.float64)

     for watermark_path, watermark_fn in watermark_tuple:
        watermark = cv2.imread(watermark_path, -1).astype(np.float64)

        for idx in range(overlay_iter):
            # if np.random.randint(0,10) % 2 == 0:
            #     output = overlay(image, watermark, (0.05, 0.18), gray=True)
            # else:
            output = overlay(image, watermark, (0.08, 0.2), gray=False)
            save_img(output, save_dir, img_fn, watermark_fn, idx)
                
def main(watermark_dir, image_dir, save_dir, overlay_iter=1):
    
    watermark_tuples = [(os.path.join(watermark_dir, cur), os.path.splitext(cur)[0]) for cur in os.listdir(watermark_dir)]
    image_tuples = [(os.path.join(image_dir, cur), os.path.splitext(cur)[0]) for cur in os.listdir(image_dir)]

    mapping = [(image_tuple, watermark_tuples, save_dir, overlay_iter) for image_tuple in  image_tuples]

    process_map(process, image_tuples, [watermark_tuples]*len(image_tuples), \
                [save_dir]*len(image_tuples), [overlay_iter]*len(image_tuples), \
                max_workers=4, chunksize=1)
    # with multiprocessing.Pool(processes=4) as pool:
    #     pool.starmap(process, mapping)

if __name__ == '__main__':
    main("./watermark", "./temp_data", "./result", overlay_iter=1)
