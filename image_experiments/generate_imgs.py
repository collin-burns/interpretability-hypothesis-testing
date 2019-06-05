"""
Given a directory of images and a directory of bounding boxes for those images, generates a new image
for each of those bounding boxes and saves in a new generated_images directory.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import maybe_resize_img, inpaint, save_mask
from inception import get_inception_probs
from os import walk

def main(args):
    img_dir = args.img_dir
    gen_dir = args.gen_dir
    bb_dir = args.bb_dir
    probs_dir = args.probs_dir
    orig_probs, inpaint_probs = [], []

    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)
    if not os.path.exists(probs_dir):
        os.mkdir(probs_dir)

    # get number of images. Assumes images are named "img{}.jpg".format(i) for i starting from 1 (NOT 0)
    for (dirpath, dirnames, filenames) in walk(img_dir):
        imgs = []
        for file in filenames:
            if file[-4:] == ".jpg":
                imgs.append(file)
        num_imgs = len(imgs)
        break

    # loop through images and preprocess, get probabilities, and inpaint
    for j in range(num_imgs):
        i = j + 1  # start from 1
        print("Starting image {}".format(i))
        img_path = os.path.join(img_dir, "img{}.jpg".format(i))
        img = plt.imread(img_path)

        # do some pre-processing to make shape of image is consistent with reshaping done by inpainting method
        # also make sure the image isn't too large
        # (Note: this resaves the image)
        img = maybe_resize_img(img, args.max_area)
        orig_shape = img.shape
        h, w, _ = orig_shape
        grid = 8
        img = img[:h//grid*grid, :w//grid*grid, :]  # As done by generative inpainting
        if img.shape != orig_shape:
            plt.imsave(img_path, img)

        # get probabilities
        probs = get_inception_probs(img)
        class_id = np.argmax(probs)
        orig_prob = get_inception_probs(img)[class_id]
        orig_probs.append(orig_prob)

        # get number of masks
        bb_img_dir = os.path.join(bb_dir, "img{}".format(i))  # bounding box subdirectory for current image
        for (dirpath, dirnames, filenames) in walk(bb_img_dir):
            num_masks = len(filenames)
            break

        cur_img_probs = []
        # now loop through masks
        for s in range(num_masks):
            k = s + 1  # starting from 1
            bb_path = os.path.join(bb_img_dir, "bb{}.npy".format(k))
            bb = np.load(bb_path)
            x_min = int(bb[0, 0])
            y_min = int(bb[0, 1])
            x_max = int(bb[1, 0])
            y_max = int(bb[1, 1])
            rec = [((x_min, y_min), (x_max, y_max))]

            # scale
            mask_file = os.path.join(gen_dir, "mask_img{}_mask{}.jpg".format(i, k))
            save_mask(rec, img.shape, mask_file)

            # inpaint
            inpainted_file = os.path.join(gen_dir, "gen_img{}_mask{}.jpg".format(i, k))
            inpainted = inpaint(img_path, mask_file, inpainted_file)  # generates, saves, and returns inpainted image

            # get prob (corresponding to predicted class on original image) for newly inpainted image
            inpainted_probs = get_inception_probs(inpainted)
            inpainted_prob = inpainted_probs[class_id]
            cur_img_probs.append(inpainted_prob)

        #save inpainted probs corresponding to current image
        cur_img_probs = np.array(cur_img_probs)
        probs_file = os.path.join(probs_dir, "img{}_probs".format(i))
        np.save(probs_file, cur_img_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_area", "-m", type=int, default=500000, help="Maximum area for an image. If greater\
                                                                           than this, image is resized and overwritten")
    parser.add_argument("--img_dir", "-i", type=str, default="sample_images")
    parser.add_argument("--gen_dir", "-g", type=str, default="generated_imgs")
    parser.add_argument("--bb_dir", "-b", type=str, default="bounding_boxes")
    parser.add_argument("--probs_dir", "-p", type=str, default="imagenet_probs")
    args = parser.parse_args()
    main(args)
