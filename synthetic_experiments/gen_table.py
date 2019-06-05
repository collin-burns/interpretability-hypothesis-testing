from __future__ import print_function
import argparse
import numpy as np
import os
import random

def get_z(stats, alpha):
    """
    For OSFT.

    Finds and returns minimum z such that (1 + # <= -z)/(1 + # >= z) <= alpha.
    Note that candidate z's, where this value actually changes, are those where z = abs(t) for some t in statistics.
    """
    flat_stats = stats.flatten()
    abs_stats = np.sort(np.abs(flat_stats))

    smallest_w = 1
    for i in range(flat_stats.shape[0]):
        z = abs_stats[i]
        num = (stats <= -z).sum() + 1
        denom = (stats >= z).sum()

        if denom == 0:
            continue

        w = num/denom
        if w < smallest_w:
            smallest_w = w
        if w <= alpha:
            # print("z: {}. w: {}. {}/{}.".format(z, w, i+1, abs_stats.shape[0]))
            # print("num rejected: {}/{}".format((stats >= z).sum(), stats.size))
            return z

    # print("Warning: failed to reject. smallest w: {}".format(smallest_w))
    return None

def get_OSFT_rejections(tstats, alpha):
    z = get_z(tstats, alpha)
    if z is not None:
        rejections = tstats >= z
    else:
        rejections = np.zeros(shape=tstats.shape, dtype=bool)
    return rejections

def get_IRT_rejections(pvals, alpha):
    """
    For IRT. Returns tau such that should reject p-values less than tau.
    Assumes pvals is has shape d by n.
    """
    flat_pvals = np.array(pvals).flatten()
    K = flat_pvals.shape[0]
    sorted_pvals = np.sort(flat_pvals)[::-1]  # decreasing order
    for i in range(K):
        k = K-i  # corresponds to the p^{(k)}
        if sorted_pvals[i] <= (k/K)*alpha:
            tau = sorted_pvals[i]
            rejections = pvals <= tau
            return rejections

def get_empirical_fdr_tpr(selected, important):
    if selected is None:
        return 0, 0

    results = important[selected]
    nselected = len(results)
    assert nselected == selected.sum()
    npos = important.sum()

    tp = results.sum()
    fp = (1-results).sum()
    tpr = tp/max(1, npos)
    fdr = fp/max(1, nselected)

    return fdr, tpr

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    ind = "independent" if args.independent else "correlated"
    disc = "disc" if args.discontinuous else "nn"
    experiment_dir = args.experiment_dir.format(ind, disc, args.experiment_num)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    print("experiment directory: ", experiment_dir)


    alphas = [0.01, 0.05, 0.1, 0.2]
    for alpha in alphas:
        print("Starting alpha: {}".format(alpha))
        directioned_IRT_fdrs = []
        directioned_IRT_tprs = []
        directionless_IRT_fdrs = []
        directionless_IRT_tprs = []
        directioned_OSFT_fdrs = []
        directioned_OSFT_tprs = []
        directionless_OSFT_fdrs = []
        directionless_OSFT_tprs = []

        for run in range(args.nruns):
            # save OSFT and IRT stats
            directioned_tstats = np.load(os.path.join(experiment_dir, "directioned_tstats{}.npy".format(run)))
            directionless_tstats = np.load(os.path.join(experiment_dir, "directionless_tstats{}.npy".format(run)))
            directioned_pvals = np.load(os.path.join(experiment_dir, "directioned_pvals{}.npy".format(run)))
            directionless_pvals = np.load(os.path.join(experiment_dir, "directionless_pvals{}.npy".format(run)))
            important = np.load(os.path.join(experiment_dir, "important{}.npy".format(run)))

            directioned_IRT_rejections = get_IRT_rejections(directioned_pvals, alpha)
            directionless_IRT_rejections = get_IRT_rejections(directionless_pvals, alpha)
            directioned_OSFT_rejections = get_OSFT_rejections(directioned_tstats, alpha)
            directionless_OSFT_rejections = get_OSFT_rejections(directionless_tstats, alpha)

            directioned_IRT_fdr, directioned_IRT_tpr = get_empirical_fdr_tpr(directioned_IRT_rejections, important)
            directionless_IRT_fdr, directionless_IRT_tpr = get_empirical_fdr_tpr(directionless_IRT_rejections, important)
            directioned_OSFT_fdr, directioned_OSFT_tpr = get_empirical_fdr_tpr(directioned_OSFT_rejections, important)
            directionless_OSFT_fdr, directionless_OSFT_tpr = get_empirical_fdr_tpr(directionless_OSFT_rejections, important)

            directioned_IRT_fdrs.append(directioned_IRT_fdr)
            directioned_IRT_tprs.append(directioned_IRT_tpr)
            directionless_IRT_fdrs.append(directionless_IRT_fdr)
            directionless_IRT_tprs.append(directionless_IRT_tpr)
            directioned_OSFT_fdrs.append(directioned_OSFT_fdr)
            directioned_OSFT_tprs.append(directioned_OSFT_tpr)
            directionless_OSFT_fdrs.append(directionless_OSFT_fdr)
            directionless_OSFT_tprs.append(directionless_OSFT_tpr)

        directioned_IRT_fdr = np.mean(directioned_IRT_fdrs)
        directioned_IRT_tpr = np.mean(directioned_IRT_tprs)
        directionless_IRT_fdr = np.mean(directionless_IRT_fdrs)
        directionless_IRT_tpr = np.mean(directionless_IRT_tprs)
        directioned_OSFT_fdr = np.mean(directioned_OSFT_fdrs)
        directioned_OSFT_tpr = np.mean(directioned_OSFT_tprs)
        directionless_OSFT_fdr = np.mean(directionless_OSFT_fdrs)
        directionless_OSFT_tpr = np.mean(directionless_OSFT_tprs)

        print("directioned IRT fdr: {}, tpr: {}".format(directioned_IRT_fdr, directioned_IRT_tpr))
        print("directionless IRT fdr: {}, tpr: {}".format(directionless_IRT_fdr, directionless_IRT_tpr))
        print("directioned OSFT fdr: {}, tpr: {}".format(directioned_OSFT_fdr, directioned_OSFT_tpr))
        print("directionless OSFT fdr: {}, tpr: {}".format(directionless_OSFT_fdr, directionless_OSFT_tpr))

        with open("all_results.txt", "a") as f:
            f.write("{}, {}, alpha: {}\n".format(ind, disc, alpha))
            f.write("directioned IRT fdr: {}, tpr: {}\n".format(directioned_IRT_fdr, directioned_IRT_tpr))
            f.write("directionless IRT fdr: {}, tpr: {}\n".format(directionless_IRT_fdr, directionless_IRT_tpr))
            f.write("directioned OSFT fdr: {}, tpr: {}\n".format(directioned_OSFT_fdr, directioned_OSFT_tpr))
            f.write("directionless OSFT fdr: {}, tpr: {}\n".format(directionless_OSFT_fdr, directionless_OSFT_tpr))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_num", "-e", type=int, default=0)  # for saving
    parser.add_argument("--experiment_dir", "-x", type=str, default="{}_{}_experiment{}")
    parser.add_argument("--seed", "-z", type=int, default=0)
    parser.add_argument("--nruns", "-n", type=int, default=10)
    parser.add_argument("--discontinuous", "-d", action="store_true")
    parser.add_argument("--independent", "-i", action="store_true")
    args = parser.parse_args()
    main(args)
