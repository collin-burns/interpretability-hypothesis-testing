import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from inception import get_inception_probs
from os import walk

def logit(prob):
    return np.log(prob/(1-prob))

def prob_diff(orig_prob, new_prob):
    return orig_prob - new_prob

def logit_diff(orig_prob, new_prob):
    return logit(orig_prob) - logit(new_prob)

def get_z(statistics, alpha):
    """
    Finds and returns minimum z such that (1 + # <= -z)/(1 + # >= z) <= alpha.
    Note that candidate z's, where this value actually changes, are those where z = abs(t) for some t in statistics.
    """
    abs_stats = np.sort(np.array([np.abs(stat) for stat in statistics]))
    stats = np.array(statistics)

    for i in range(abs_stats.shape[0]):
        z = abs_stats[i]
        num = (stats <= -z).sum() + 1
        denom = (stats >= z).sum() + 1
        if denom == 0:
            w = 0
        else:
            w = num/denom
        if w <= alpha:
            print("z: {}. w: {}. {}/{}.".format(z, w, i+1, abs_stats.shape[0]))
            print("num rejected: {}/{}".format((stats >= z).sum(), stats.size))
            return z

    print("Warning: failed to reject.")
    return None


def main(args):
    for (_, _, probs_files) in walk(args.probs_dir):
        num_imgs = len(probs_files)
        break

    if args.test_statistic == "prob_diff":
        statistic = prob_diff
    else:
        statistic = logit_diff

    if not args.load:
        # now get a list of all the test statistics
        statistics = []
        img_stats = []  # list of list, for later use
        for i in range(num_imgs):
            print("starting image {}/{}".format(i+1, num_imgs))
            inpainted_probs = np.load(os.path.join(args.probs_dir, "img{}_probs.npy".format(i+1)))  # inpainted probs
            img = plt.imread(os.path.join(args.img_dir, "img{}.jpg".format(i+1)))
            orig_probs = get_inception_probs(img)
            class_id = np.argmax(orig_probs)
            orig_prob = orig_probs[class_id]

            # loop over number of masks
            cur_img_stats = []
            for j in range(inpainted_probs.shape[0]):
                inpainted_prob = inpainted_probs[j]
                stat = statistic(orig_prob, inpainted_prob)
                statistics.append(stat)
                cur_img_stats.append(stat)
            img_stats.append(cur_img_stats)
        np.save("{}_stats.npy".format(args.test_statistic), np.array(statistics))
    else:
        statistics = np.load("{}_stats.npy".format(args.test_statistic))

    # determine the z value (threshold for rejection)
    z = get_z(statistics, args.alpha)

    # output all of the rejections, and save to output file
    output_file = args.output_file.format(args.test_statistic)
    with open(output_file, "w") as f:
        # write the rejection threshold
        f.write("{}\n".format(z))
        for i, stats in enumerate(img_stats):
            for j, stat in enumerate(stats):
                # print {img_num},{mask_num},{w} on line in txt file
                f.write("{},{},{}\n".format(i+1, j+1, stat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs_dir", "-p", type=str, default="imagenet_probs")
    parser.add_argument("--img_dir", "-i", type=str, default="sample_images")
    parser.add_argument("--test_statistic", "-t", type=str, default="logit_diff")
    parser.add_argument("--alpha", "-a", type=float, default=0.2)
    parser.add_argument("--output_file", "-o", type=str, default="{}_rejections.txt")
    parser.add_argument("--load", "-l", type=bool, default=False)
    args = parser.parse_args()
    main(args)