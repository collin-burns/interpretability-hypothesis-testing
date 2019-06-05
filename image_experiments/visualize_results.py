import argparse
import os
import numpy as np
from inception import get_inception_probs
from hypothesis_test import prob_diff, logit_diff
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from id_to_name import id_to_name

def main(args):
    """
    For each image, viusalize the bounding boxes being tested, where the box is blue if selected and red otherwise
    """
    save_dir = args.save_dir.format(args.test_statistic)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.test_statistic == "prob_diff":
        statistic = prob_diff
    else:
        statistic = logit_diff
    with open(args.reject_file.format(args.test_statistic)) as f:
        for i, l in enumerate(f):
            pass
    final_i = i

    with open(args.reject_file.format(args.test_statistic), "r") as f:
        prev_img_num = 0
        for i, line in enumerate(f):
            if i == 0:  # first line of reject file is the rejection threshold
                reject_w = float(line[:-1])
                continue

            # if it's the last line of the final, save the image
            if i == final_i:
                probs = get_inception_probs(prev_img)
                predicted_id = np.argmax(probs)
                name = id_to_name[predicted_id]
                orig_prob = probs[predicted_id]
                ax.set_title("w threshold: {:.4f}\n{} (p: {:3f})".format(reject_w, name, orig_prob))

                fig.tight_layout()
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                plt.savefig(os.path.join(save_dir, "img{}_results.png".format(prev_img_num)), bbox_inches='tight', pad_inches=0)
                plt.close()
                break

            # parse line
            loc1 = line.find(",")
            img_num = int(line[:loc1])
            remaining = line[loc1+1:]
            loc2 = remaining.find(",")
            mask_num = int(remaining[:loc2])
            w = float(remaining[loc2+1:-1])

            # now load original image
            orig_img = plt.imread(os.path.join(args.img_dir, "img{}.jpg".format(img_num)))

            # if first image, start figure
            if i == 1:  # since skipped first line
                fig, ax = plt.subplots(1)
                ax.imshow(orig_img)

            # if new image (but not first image), save old image and start new one
            if prev_img_num != img_num and prev_img_num != 0:
                probs = get_inception_probs(prev_img)
                predicted_id = np.argmax(probs)
                name = id_to_name[predicted_id]
                orig_prob = probs[predicted_id]
                ax.set_title("w threshold: {:.4f}\n{} (p: {:3f})".format(reject_w, name, orig_prob))
                fig.tight_layout()
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                plt.savefig(os.path.join(save_dir, "img{}_results.pdf".format(prev_img_num)), bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                # show new image
                fig, ax = plt.subplots(1)
                ax.imshow(orig_img)

            # show new bounding box
            if w >= reject_w:
                edgecolor = "blue"
            else:
                edgecolor = "red"

            # load bounding box
            bb = np.load(os.path.join(args.bb_dir, "img{}".format(img_num), "bb{}.npy".format(mask_num)))
            x_min = bb[0, 0]
            y_min = bb[0, 1]
            x_max = bb[1, 0]
            y_max = bb[1, 1]

            # put it in the right format for patches.Rectangle
            x_left = x_min
            y_left = y_min
            width = x_max-x_min
            height = y_max-y_min

            # show rectangle
            rect = patches.Rectangle((x_left, y_left), width, height, linewidth=2, edgecolor=edgecolor, facecolor="none")
            ax.add_artist(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax.annotate("{:.4f}".format(w), (cx, cy), color='w', weight='bold',
                fontsize=8, ha='center', va='center')

            prev_img_num = img_num
            prev_img = orig_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", "-i", type=str, default="sample_images")
    parser.add_argument("--save_dir", "-s", type=str, default="{}_results")
    parser.add_argument("--gen_dir", "-g", type=str, default="generated_imgs")
    parser.add_argument("--bb_dir", "-b", type=str, default="bounding_boxes")
    parser.add_argument("--reject_file", "-r", type=str, default="{}_rejections.txt")
    parser.add_argument("--test_statistic", "-t", type=str, default="logit_diff")
    args = parser.parse_args()
    main(args)
