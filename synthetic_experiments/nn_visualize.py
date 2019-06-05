import numpy as np
import argparse
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import matplotlib

TPR_IDX = 0
FDR_IDX = 1
FPR_IDX = 2

# Order: IRT, OSFT, LIME, SHAP, grad*in, saliency, DeepLIFT, L2X
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
linestyles = ["-", "-", "-", "-", "-", "-", "--", "-"]
lws = [5, 5, 3, 3, 3, 3, 3, 3]

def get_top_k_features(feature_values, k):
    """
    Works for shap, lime, and the saliency map methods. assumes feature_values has shape nsamples x nfeatures, where
    features_values[i, j] is the value assigned to the jth feature of sample i by either shap or lime.

    Returns exactly k top features, even if some have the same exact value by breaking ties randomly.

    returns a mask corresponding to the top k features (out of nsamples*nfeatures total, NOT just for one sample)

    assumes k >= 0 is an int
    """
    flat_vals = feature_values.ravel()
    flat_vals += 1E-18*np.random.random(size=flat_vals.shape)  # for breaking ties

    if k != 0:
        kth_largest_idx = np.argsort(flat_vals)[-k]
        kth_largest_val = flat_vals[kth_largest_idx]
        selected_features_mask = (feature_values >= kth_largest_val)
    else:
        selected_features_mask = np.zeros(shape=feature_values.shape, dtype=bool)

    return np.squeeze(selected_features_mask)

def get_IRT_stats(pvals, important):
    p = pvals.flatten()
    # k = 0 done manually
    fdrs = [0]
    fprs = [0]
    tprs = [0]

    p_orders = np.argsort(p)[::-1]  # indexes corresponding to p-values in decreasing order
    for i, s in enumerate(p_orders):
        k = i + 1
        discoveries = np.zeros(p.shape, dtype=int)
        discoveries[p_orders[-k:]] = 1  # take the smallest k p-values

        pselected = np.where(discoveries == 1)[0]
        results = important.flatten()[pselected]
        nselected = discoveries.sum()
        assert nselected == len(results)
        ntested = len(p)

        npos = important.sum()
        nneg = ntested - npos
        fp = (1-results).sum()
        tp = results.sum()
        tpr = tp / max(1, npos)
        fdr = fp / max(1, nselected)
        fpr = fp / max(1, nneg)
        tprs.append(tpr)
        fdrs.append(fdr)
        fprs.append(fpr)
    return np.array((tprs, fdrs, fprs))

def get_osft_stats(tstats, important):
    m = (~np.isnan(tstats))
    tstats_flat = tstats[m]
    fdrs = []
    fprs = []
    tprs = []

    # sort such that ties are broken randomly (using lexsort), then take the largest k t-statistics for increasing k
    idxs_ordered = np.lexsort((np.random.random(tstats_flat.size), tstats_flat))[::-1]
    for k in range(idxs_ordered.size + 1):  # want to include both k=0 and k=idxs_ordered.size (i.e., max #features)
        max_k_idxs = idxs_ordered[:k]  # selected tstats
        results = important[m][max_k_idxs]  # among those selected, which are important (T/F)

        nselected = len(results)
        npos = important[m].sum()
        nneg = (1-important[m]).sum()
        tp = results.sum()
        fp = (1-results).sum()
        tpr = tp / max(1, npos)
        fdr = fp / max(1, nselected)
        fpr = fp / max(1, nneg)
        tprs.append(tpr)
        fdrs.append(fdr)
        fprs.append(fpr)
    return np.array((tprs, fdrs, fprs))

def get_val_stats(vals, important):
    """
    Get the statistics (tprs, fdrs, and fprs) for either shapley or lime (since both similarly output feature values).
    """
    max_ntest = vals.size
    fprs = []
    fdrs = []
    tprs = []

    for k in range(max_ntest + 1):  # +1 so that tests both 0 features and max_ntest features
        selected_features = get_top_k_features(vals, k)

        results = important[selected_features]
        nselected = len(results)
        npos = important.sum()
        nneg = (1-important).sum()

        tp = results.sum()
        fp = (1-results).sum()
        tpr = tp/max(1, npos)
        fdr = fp/max(1, nselected)
        fpr = fp/max(1, nneg)

        fprs.append(fpr)
        fdrs.append(fdr)
        tprs.append(tpr)
    return np.array((tprs, fdrs, fprs))

def get_l2x_stats(scores, important):
    """
    Because l2x necessarily selects k features per sample, and only outputs binary scores (instead of real feature
    values like lime and shap) it is necessary to vary k from 0 to nfeatures, making this less fine grained than
    the other comparisons.

    Note that scores has shape nfeatures x nsamples x nfeatures, with scores[i] corresponding to the scores
    when k = i-1 features are selected per sample.
    """
    # do k=0 manually
    fprs = [0]
    fdrs = [0]
    tprs = [0]
    nfeatures = scores.shape[2]
    max_ntest = nfeatures

    for i in range(max_ntest):
        k = i + 1
        selected_features = (scores[k-1] != 0)

        results = important[selected_features]
        nselected = len(results)
        assert nselected == selected_features.sum()
        npos = important.sum()
        nneg = (1-important).sum()

        tp = results.sum()
        fp = (1-results).sum()
        tpr = tp/max(1, npos)
        fdr = fp/max(1, nselected)
        fpr = fp/max(1, nneg)

        fprs.append(fpr)
        fdrs.append(fdr)
        tprs.append(tpr)
    return np.array((tprs, fdrs, fprs))

def fix(xs, ys):
    """
    For each x in xs, lets new_y be the largest value in ys such that new_x corresponding to new_y is at most x.
    """
    new_xs = np.arange(0, 1.01, 0.01)
    new_ys = [0]

    for j in range(new_xs.shape[0]-1):
        i = j + 1  # Skip 0 (add y=0 manually)
        x = new_xs[i]
        new_y = np.max(ys[xs <= x])
        new_ys.append(new_y)
    new_ys = np.array(new_ys)

    return new_xs, new_ys

def identity(xs, ys):
    return xs, ys

def main(args):
    ind = "independent" if args.independent else "correlated"
    experiment_dir = args.experiment_dir.format(ind, args.experiment_num)

    if args.no_smooth:
        process = identity
    else:
        process = fix

    if not args.load:
        two_gradin_stats = []
        one_gradin_stats = []
        two_saliency_stats = []
        one_saliency_stats = []
        two_deeplift_stats = []
        one_deeplift_stats = []
        two_osft_stats = []  # two-sided (undirected)
        one_osft_stats = []  # one-sided
        two_irt_stats = []
        one_irt_stats = []
        two_shap_stats = []
        one_shap_stats = []
        two_lime_stats = []
        one_lime_stats = []
        l2x_stats = []
        for run in range(args.nruns):
            print("Starting run {}/{}".format(run+1, args.nruns))
            important = np.load(os.path.join(experiment_dir, "important{}.npy".format(run)))
            print("fraction important", important.mean())

            print("starting grad*input")
            gradin_vals = np.load(os.path.join(experiment_dir, "gradinput{}.npy".format(run)))
            two_gradin_vals = np.abs(gradin_vals)
            two_gradin_stats.append(get_val_stats(two_gradin_vals, important))
            one_gradin_stats.append(get_val_stats(gradin_vals, important))

            print("starting saliency")
            saliency_vals = np.load(os.path.join(experiment_dir, "saliency{}.npy".format(run)))
            two_saliency_vals = np.abs(saliency_vals)
            two_saliency_stats.append(get_val_stats(two_saliency_vals, important))
            one_saliency_stats.append(get_val_stats(saliency_vals, important))

            print("starting deeplift")
            deeplift_vals = np.load(os.path.join(experiment_dir, "deeplift{}.npy".format(run)))
            two_deeplift_vals = np.abs(deeplift_vals)
            two_deeplift_stats.append(get_val_stats(two_deeplift_vals, important))
            one_deeplift_stats.append(get_val_stats(deeplift_vals, important))

            print("starting OSFT")
            # osft
            directioned_tstats = np.load(os.path.join(experiment_dir, "directioned_tstats{}.npy".format(run)))
            directionless_tstats = np.load(os.path.join(experiment_dir, "directionless_tstats{}.npy".format(run)))
            two_osft_stats.append(get_osft_stats(directionless_tstats, important))
            one_osft_stats.append(get_osft_stats(directioned_tstats, important))

            print("starting IRT")
            # IRT
            directionless_pvals = np.load(os.path.join(experiment_dir, "directionless_pvals{}.npy".format(run)))
            directioned_pvals = np.load(os.path.join(experiment_dir, "directioned_pvals{}.npy".format(run)))
            two_irt_stats.append(get_IRT_stats(directionless_pvals, important))
            one_irt_stats.append(get_IRT_stats(directioned_pvals, important))

            print("starting SHAP")
            # SHAP
            shap_vals = np.load(os.path.join(experiment_dir, "shap_vals{}.npy".format(run)))
            two_shap_vals = np.abs(shap_vals)
            two_shap_stats.append(get_val_stats(two_shap_vals, important))
            one_shap_stats.append(get_val_stats(shap_vals, important))

            if not args.skip_lime:
                print("starting lime")
                # LIME
                lime_vals = np.load(os.path.join(experiment_dir, "lime_vals{}.npy".format(run)))
                two_lime_vals = np.abs(lime_vals)
                one_lime_stats.append(get_val_stats(lime_vals, important))
                two_lime_stats.append(get_val_stats(two_lime_vals, important))

            if not args.skip_l2x:
                print("starting l2x")
                # L2X
                l2x_scores = np.load(os.path.join(experiment_dir, "l2x_scores{}.npy".format(run)))
                l2x_stats.append(get_l2x_stats(l2x_scores, important))

        # save these statistics so don't have to recompute
        np.save(os.path.join(experiment_dir, "one_gradin_stats.npy"), one_gradin_stats)
        np.save(os.path.join(experiment_dir, "two_gradin_stats.npy"), two_gradin_stats)
        np.save(os.path.join(experiment_dir, "one_saliency_stats.npy"), one_saliency_stats)
        np.save(os.path.join(experiment_dir, "two_saliency_stats.npy"), two_saliency_stats)
        np.save(os.path.join(experiment_dir, "one_deeplift_stats.npy"), one_deeplift_stats)
        np.save(os.path.join(experiment_dir, "two_deeplift_stats.npy"), two_deeplift_stats)
        np.save(os.path.join(experiment_dir, "one_osft_stats.npy"), one_osft_stats)
        np.save(os.path.join(experiment_dir, "two_osft_stats.npy"), two_osft_stats)
        np.save(os.path.join(experiment_dir, "one_irt_stats.npy"), one_irt_stats)
        np.save(os.path.join(experiment_dir, "two_irt_stats.npy"), two_irt_stats)
        np.save(os.path.join(experiment_dir, "one_shap_stats.npy"), one_shap_stats)
        np.save(os.path.join(experiment_dir, "two_shap_stats.npy"), two_shap_stats)
        np.save(os.path.join(experiment_dir, "one_lime_stats.npy"), one_lime_stats)
        np.save(os.path.join(experiment_dir, "two_lime_stats.npy"), two_lime_stats)
        if not args.skip_l2x:
            np.save(os.path.join(experiment_dir, "l2x_stats.npy"), l2x_stats)
    else:  # load
        two_gradin_stats = np.load(os.path.join(experiment_dir, "two_gradin_stats.npy"))
        one_gradin_stats = np.load(os.path.join(experiment_dir, "one_gradin_stats.npy"))
        two_saliency_stats = np.load(os.path.join(experiment_dir, "two_saliency_stats.npy"))
        one_saliency_stats = np.load(os.path.join(experiment_dir, "one_saliency_stats.npy"))
        two_deeplift_stats = np.load(os.path.join(experiment_dir, "two_deeplift_stats.npy"))
        one_deeplift_stats = np.load(os.path.join(experiment_dir, "one_deeplift_stats.npy"))
        two_osft_stats = np.load(os.path.join(experiment_dir, "two_osft_stats.npy"))
        one_osft_stats = np.load(os.path.join(experiment_dir, "one_osft_stats.npy"))
        two_irt_stats = np.load(os.path.join(experiment_dir, "two_irt_stats.npy"))
        one_irt_stats = np.load(os.path.join(experiment_dir, "one_irt_stats.npy"))
        two_shap_stats = np.load(os.path.join(experiment_dir, "two_shap_stats.npy"))
        one_shap_stats = np.load(os.path.join(experiment_dir, "one_shap_stats.npy"))
        two_lime_stats = np.load(os.path.join(experiment_dir, "two_lime_stats.npy"))
        one_lime_stats = np.load(os.path.join(experiment_dir, "one_lime_stats.npy"))
        if not args.skip_l2x:
            l2x_stats = np.load(os.path.join(experiment_dir, "l2x_stats.npy"))

    axis_size = 22
    tick_size = 16
    legend_font_size = 13

    mean_two_gradin = np.mean(two_gradin_stats, axis=0)
    mean_one_gradin = np.mean(one_gradin_stats, axis=0)
    mean_two_saliency = np.mean(two_saliency_stats, axis=0)
    mean_one_saliency = np.mean(one_saliency_stats, axis=0)
    mean_two_deeplift = np.mean(two_deeplift_stats, axis=0)
    mean_one_deeplift = np.mean(one_deeplift_stats, axis=0)
    mean_two_osft = np.mean(two_osft_stats, axis=0)
    mean_one_osft = np.mean(one_osft_stats, axis=0)
    mean_two_irt = np.mean(two_irt_stats, axis=0)
    mean_one_irt = np.mean(one_irt_stats, axis=0)
    mean_two_shap = np.mean(two_shap_stats, axis=0)
    mean_one_shap = np.mean(one_shap_stats, axis=0)
    if not args.skip_lime:
        mean_two_lime = np.mean(two_lime_stats, axis=0)
        mean_one_lime = np.mean(one_lime_stats, axis=0)
    if not args.skip_l2x:
        mean_l2x = np.mean(l2x_stats, axis=0)

    with sns.axes_style('white'):
        # one sided
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        xs, ys = process(mean_one_irt[FDR_IDX], mean_one_irt[TPR_IDX])
        plt.plot(xs, ys, label="IRT", color=colors[0], linestyle=linestyles[0], lw=lws[0])
        xs, ys = process(mean_one_osft[FDR_IDX], mean_one_osft[TPR_IDX])
        plt.plot(xs, ys, label="OSFT", color=colors[1], linestyle=linestyles[1], lw=lws[1])
        if not args.skip_lime:
            xs, ys = process(mean_one_lime[FDR_IDX], mean_one_lime[TPR_IDX])
            plt.plot(xs, ys, label="LIME", color=colors[2], linestyle=linestyles[2], lw=lws[2])
        xs, ys = process(mean_one_shap[FDR_IDX], mean_one_shap[TPR_IDX])
        plt.plot(xs, ys, label="SHAP", color=colors[3], linestyle=linestyles[3], lw=lws[3])
        xs, ys = process(mean_one_gradin[FDR_IDX], mean_one_gradin[TPR_IDX])
        plt.plot(xs, ys, label="Taylor", color=colors[4], linestyle=linestyles[4], lw=lws[4])
        xs, ys = process(mean_one_saliency[FDR_IDX], mean_one_saliency[TPR_IDX])
        plt.plot(xs, ys, label="Saliency", color=colors[5], linestyle=linestyles[5], lw=lws[5])
        xs, ys = process(mean_one_deeplift[FDR_IDX], mean_one_deeplift[TPR_IDX])
        plt.plot(xs, ys, label="DeepLIFT", color=colors[6], linestyle=linestyles[6], lw=lws[6])

        plt.xlabel("FDR", fontsize=axis_size)
        plt.ylabel("TPR", fontsize=axis_size)
        plt.axis('scaled')
        plt.grid()

        legend_props = {'weight': 'bold', 'size': legend_font_size}
        if args.include_title:
            plt.title("One-sided. Mean over {} runs.".format(args.nruns))
        plt.tight_layout()
        if args.include_legend:
            plt.legend(loc='lower right', prop=legend_props)
            plt.savefig(os.path.join(experiment_dir, "one_sided_legend.pdf"), bbox_inches="tight")
        else:
            plt.savefig(os.path.join(experiment_dir, "one_sided.pdf"), bbox_inches="tight")
        plt.close()

        # two sided
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        xs, ys = process(mean_two_irt[FDR_IDX], mean_two_irt[TPR_IDX])
        plt.plot(xs, ys, label="IRT", color=colors[0], linestyle=linestyles[0], lw=lws[0])
        xs, ys = process(mean_two_osft[FDR_IDX], mean_two_osft[TPR_IDX])
        plt.plot(xs, ys, label="OSFT", color=colors[1], linestyle=linestyles[1], lw=lws[1])
        if not args.skip_lime:
            xs, ys = process(mean_two_lime[FDR_IDX], mean_two_lime[TPR_IDX])
            plt.plot(xs, ys, label="LIME", color=colors[2], linestyle=linestyles[2], lw=lws[2])
        xs, ys = process(mean_two_shap[FDR_IDX], mean_two_shap[TPR_IDX])
        plt.plot(xs, ys, label="SHAP", color=colors[3], linestyle=linestyles[3], lw=lws[3])

        print("printing Taylor. mean: {}".format(mean_two_gradin[TPR_IDX].mean()))
        xs, ys = process(mean_two_gradin[FDR_IDX], mean_two_gradin[TPR_IDX])
        plt.plot(xs, ys, label="Taylor", color=colors[4], linestyle=linestyles[4], lw=lws[4])

        xs, ys = process(mean_two_saliency[FDR_IDX], mean_two_saliency[TPR_IDX])
        plt.plot(xs, ys, label="Saliency", color=colors[5], linestyle=linestyles[5], lw=lws[5])
        xs, ys = process(mean_two_deeplift[FDR_IDX], mean_two_deeplift[TPR_IDX])
        plt.plot(xs, ys, label="DeepLIFT", color=colors[6], linestyle=linestyles[6], lw=lws[6])
        if not args.skip_l2x:
            xs, ys = process(mean_l2x[FDR_IDX], mean_l2x[TPR_IDX])
            plt.plot(xs, ys, label="L2X", color=colors[7], linestyle=linestyles[7], lw=lws[7])

        plt.xlabel("FDR", fontsize=axis_size)
        plt.ylabel("TPR", fontsize=axis_size)
        plt.axis('scaled')
        plt.grid()
        legend_props = {'weight': 'bold', 'size': legend_font_size}
        if args.include_title:
            plt.title("Two-sided. Mean over {} runs.".format(args.nruns))
        plt.tight_layout()
        if args.include_legend:
            plt.legend(loc='lower right', prop=legend_props)
            plt.savefig(os.path.join(experiment_dir, "two_sided_legend.pdf"), bbox_inches="tight")
        else:
            plt.savefig(os.path.join(experiment_dir, "two_sided.pdf"), bbox_inches="tight")
        plt.close()

    # also get the largest TPR for alpha = 0.05, 0.1, 0.2 for each of these methods
    alphas = [0.05, 0.1, 0.2]
    two_sided = [mean_two_gradin, mean_two_saliency, mean_two_deeplift, mean_two_irt, mean_two_osft, mean_two_shap]
    one_sided = [mean_one_gradin, mean_one_saliency, mean_one_deeplift, mean_one_irt, mean_one_osft, mean_one_shap]
    names = ["grad*input", "Saliency", "DeepLIFT", "IRT", "OSFT", "SHAP"]

    if not args.skip_lime:
        two_sided.append(mean_two_lime)
        one_sided.append(mean_one_lime)
        names.append("LIME")
    if not args.skip_l2x:
        two_sided.append(mean_l2x)
        one_sided.append(mean_l2x)
        names.append("L2X")

    with open(os.path.join(experiment_dir, "results.txt"), "w") as f:
        print("")
        print("TWO SIDED STATS")
        f.write("two sided stats\n")
        for alpha in alphas:
            for stats, name in zip(two_sided, names):
                tprs = stats[TPR_IDX]
                fdrs = stats[FDR_IDX]
                max_tpr = np.max(tprs[fdrs < alpha])
                print("{}, fdr {}, tpr {:.3f}".format(name, alpha, max_tpr))
                f.write("{}, fdr {}, tpr {:.3f}\n".format(name, alpha, max_tpr))
        print("")
        print("ONE SIDED STATS")
        f.write("one sided stats\n")
        for alpha in alphas:
            for stats, name in zip(one_sided, names):
                tprs = stats[TPR_IDX]
                fdrs = stats[FDR_IDX]
                max_tpr = np.max(tprs[fdrs < alpha])
                print("{}, fdr {}, tpr {:.3f}".format(name, alpha, max_tpr))
                f.write("{}, fdr {}, tpr {:.3f}\n".format(name, alpha, max_tpr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", "-d", type=str, default="{}_nn_experiment{}")
    parser.add_argument("--experiment_num", "-e", type=int, default=0)
    parser.add_argument("--load", "-o", action="store_true")
    parser.add_argument("--nruns", "-r", type=int, default=10)
    parser.add_argument("--skip_l2x", "-s", action="store_true")
    parser.add_argument("--skip_lime", "-l", action="store_true")
    parser.add_argument("--include_title", "-t", action="store_true")
    parser.add_argument("--include_legend", "-n", action="store_true")
    parser.add_argument("--no_smooth", "-m", action="store_true")
    parser.add_argument("--independent", "-i", action="store_true")
    args = parser.parse_args()
    main(args)
