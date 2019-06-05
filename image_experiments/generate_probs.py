"""
Run if generate_probs was ran without generating the probabilities. Assumes all of the
inpainted images have been generated already.
"""
import os
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import subprocess
from utilities import *
from inception import get_inception_probs
from os import walk


def main(args):
    img_dir = args.img_dir
    gen_dir = args.gen_dir
    bb_dir = args.bb_dir
    probs_dir = args.probs_dir

    orig_probs, inpaint_probs = [], []

    # get number of images. Assumes images are named "img{}.jpg".format(i) for i STARTING FROM 1
    for (dirpath, dirnames, filenames) in walk(img_dir):
        imgs = []
        for file in filenames:
            if file[-4:] == ".jpg":
                imgs.append(file)
        num_imgs = len(imgs)
        print(num_imgs)
        break

    # loop through images and preprocess, get probabilities, and inpaint
    for j in range(num_imgs):
        i = j + args.start_num
        print("Starting image {}".format(i))
        img_path = os.path.join(img_dir, "img{}.jpg".format(i))
        img = plt.imread(img_path)

        # get probabilities
        probs = get_inception_probs(img)
        class_id = np.argmax(probs)
        orig_prob = get_inception_probs(img)[class_id]
        orig_probs.append(orig_prob)
        print("class id: {}, orig prob: {}".format(class_id, orig_prob))

        # get number of masks
        bb_img_dir = os.path.join(bb_dir, "img{}".format(i))  # bounding boxes for current image
        for (dirpath, dirnames, filenames) in walk(bb_img_dir):
            num_masks = len(filenames)
            break

        cur_img_probs = []
        # now loop through masks
        for s in range(num_masks):
            k = s + 1
            # now inpaint
            inpainted_file = os.path.join(gen_dir, "gen_img{}_mask{}.jpg".format(i, k))

            # now get prob (corresponding to predicted class on original image) for newly inpainted image
            inpainted = plt.imread(inpainted_file)
            inpainted_probs = get_inception_probs(inpainted)
            inpainted_prob = inpainted_probs[class_id]
            print("inpainted prob: {}. orig prob: {}".format(inpainted_prob, orig_prob))
            cur_img_probs.append(inpainted_prob)

        # save inpainted probs corresponding to current image
        cur_img_probs = np.array(cur_img_probs)
        probs_file = os.path.join(probs_dir, "img{}_probs".format(i))
        np.save(probs_file, cur_img_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_area", "-m", type=int, default=500000, help="Maximum area for an image. If greater\
                                                                           than this, image is resized and overwritten")
    parser.add_argument("--img_dir", "-i", type=str, default="sample_imagenet_images")
    parser.add_argument("--gen_dir", "-g", type=str, default="generated_imgs")
    parser.add_argument("--bb_dir", "-b", type=str, default="bounding_boxes")
    parser.add_argument("--probs_dir", "-p", type=str, default="imagenet_probs")
    parser.add_argument("--start_num", "-n", type=int, default=1)
    args = parser.parse_args()
    main(args)
