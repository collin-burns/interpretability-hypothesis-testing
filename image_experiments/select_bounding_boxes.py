"""
Given an image and bounding box locations, saves the bounding boxes as numpy arrays.
This facilitates choosing the bounding boxes by hand. You can determine the coordinates using
a program like Photoshop or Gimp.

Usage: python select_bounding_boxes.py [img number]
By default, saves to the directory bounding_boxes
"""
import os
import argparse
import numpy as np

def save_bb(x1, y1, x2, y2, save_num, save_dir):
    """
    Saves the given bounding box as a numpy file that will be used later.
    Assumes x1 < x2 and y1 < y2.
    """
    bbs = np.array([[x1, y1], [x2, y2]])
    save_path = os.path.join(save_dir, "bb{}".format(save_num))
    np.save(save_path, bbs)

def main(img_num, bb_dir):
    if not os.path.exists(bb_dir):
        os.mkdir(bb_dir)

    save_dir = os.path.join(bb_dir, "img{}".format(img_num))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    while True:
        x1 = float(input("x1: "))
        y1 = float(input("y1: "))
        x2 = float(input("x2: "))
        y2 = float(input("y2: "))
        save_num = int(input("save num: "))
        save_bb(x1, y1, x2, y2, save_num, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_num", type=str)
    parser.add_argument("--bb_dir", "-d", type=str, default="bounding_boxes")
    args = parser.parse_args()
    main(args.img_num, args.bb_dir)