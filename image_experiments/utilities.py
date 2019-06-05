import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import subprocess

def inpaint(img_path, mask_path, inpainted_path):
    """
    Given a path to an image (img_path) and mask (mask_path), runs a subprocess calling the generative inpainter
    to infill the model for that mask. It then saves the result to inpainted_path and returns the generated image.

    Assumes that generative_inpainting has been cloned into the parent directory of image_experiments.
    Assumes that the imagenet pretrained model has been downloaded (see the GI repo for instructions).
    """
    inpainted_path = os.path.abspath(inpainted_path)
    mask_path = os.path.abspath(mask_path)
    inpaint_model_path = os.path.abspath("../generative_inpainting/test.py")
    checkpoint = os.path.abspath("../generative_inpainting/model_logs/release_imagenet_256")
    command = "python {} --image {} --mask {} --output {} --checkpoint_dir {}".format(inpaint_model_path, img_path,
                                                                                      mask_path, inpainted_path,
                                                                                      checkpoint)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)

    inpainted = plt.imread(inpainted_path)
    return inpainted

def scale_recs(img_shape, recs):
    """
    Resizes bounding boxes.
    """
    y_length, x_length, _ = img_shape
    scaled_recs = []
    for rec in recs:
        ((x_min, y_min), (x_max, y_max)) = rec
        x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
        scaled_x_min, scaled_x_max = int(np.round(x_min*x_length)), int(np.round(x_max*x_length))
        scaled_y_min, scaled_y_max = int(np.round(y_min*y_length)), int(np.round(y_max*y_length))

        scaled_recs.append([(scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max)])
    return scaled_recs

def save_mask(scaled_recs, img_shape, mask_path):
    """
    Given scaled bounding box (scaled_recs), saves to mask_path to be compatible with GI.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    for rec in scaled_recs:
        [(x_min, y_min), (x_max, y_max)] = rec
        mask[y_min:y_max, x_min:x_max, :] = 255
    plt.imsave(mask_path, mask)

def maybe_resize_img(img, max_area):
    """
    If the given image has area larger than max_area, this function resizes the image to have area under
    max_area, while retaining the same aspect ratio.
    """
    area = img.shape[0]*img.shape[1]
    if area <= max_area:
        return img
    resize_factor = max_area/area
    img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    assert img.shape[0]*img.shape[1] < max_area
    return img
