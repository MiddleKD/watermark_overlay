import cv2
import numpy as np
import os

target_watermark_dir = "./watermark"
sample_path = "./result/sample.jpg"

watermark_patterns = [cv2.imread(os.path.join(target_watermark_dir, cur), -1) for cur in os.listdir(target_watermark_dir)]
photo = cv2.imread(sample_path)


for idx, pattern in enumerate(watermark_patterns):
    # Calculate how many copies of the pattern will be needed in each direction
    tile_x = photo.shape[1] // pattern.shape[1] + 1
    tile_y = photo.shape[0] // pattern.shape[0] + 1

    # Create a tiled pattern that is larger than the photo
    tiled_pattern = np.tile(pattern, (tile_y, tile_x, 1))

    # Crop the tiled pattern to match the photo size
    tiled_pattern = tiled_pattern[:photo.shape[0], :photo.shape[1]]

    # Split the pattern into BGR and alpha channels
    pattern_bgr = tiled_pattern[..., :3]
    pattern_alpha = tiled_pattern[..., 3]

    # Create an alpha mask with three identical channels, normalized to range 0..1
    alpha_mask = cv2.merge([pattern_alpha, pattern_alpha, pattern_alpha]) / 255.0 * 0.1

    # Perform the overlay: dst = src1*alpha + src2*(1-alpha)
    overlay = pattern_bgr * alpha_mask + photo * (1 - alpha_mask)

    # Convert overlay from float to uint8
    overlay = overlay.astype(np.uint8)

    # Save the result
    cv2.imwrite(f"./result/overlay_{idx}.png", overlay)