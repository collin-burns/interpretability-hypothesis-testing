#!/usr/bin/env python3
# Takes in the JSON mask extension eval results of run_classifier.py for analysis.

import argparse
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import sys


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sentence_str(tks):
    return ' '.join([tk.encode('ascii', 'ignore').decode('ascii') for tk in tks if tk != "[PAD]"])


def main(args):
    assert args.test in ['directed', 'directionless', 'directionless-argmax']
    assert args.logit_diff == -1 or args.nlabels == 2

    print("Loading data from '%s'..." % args.input_fn)
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
    print("... done; got %d inputs" % len(d["inputs"]))

    print("Counting source sentences and expected rows and performance CM...")
    num_source = 0
    total_rows = 0  # we will have rows for all words, even those that the in-fill model matched that don't appear here
    ids_to_idxs = {}  # map stored maskdata ids to json data idxs for source sentences
    ids_to_nrows = {}  # map stored maskdata ids to expected number of rows
    max_row_by_sentence = 0
    cm = [[0 for idx in range(args.nlabels)] for jdx in range(args.nlabels)]
    for idx in range(len(d["inputs"])):
        if d["maskdata"][idx][1] == -1:  # source sentence
            sid = d["maskdata"][idx][0]
            if sid not in ids_to_nrows:
                ids_to_nrows[sid] = 0
            num_source += 1
            label = d["labels"][idx]
            for jdx in range(len(d["inputs"][idx])):
                if d["inputs"][idx][jdx] != "[PAD]":
                    total_rows += 1
                    ids_to_nrows[sid] += 1
                else:
                    break
            max_row_by_sentence = max(max_row_by_sentence, ids_to_nrows[sid])
            ids_to_idxs[d["maskdata"][idx][0]] = idx
            pred = int(np.argmax(d["logits"][idx]))
            cm[label][pred] += 1
    print("... done; %d sentences of length %d producing up to %d samples" %
          (num_source, max_row_by_sentence, num_source * max_row_by_sentence))
    print("... confusion matrix for task:")
    print("\t\t" + '\t'.join(["L%d" % idx for idx in range(args.nlabels)]))
    for jdx in range(args.nlabels):
        print("\t" + "P%d" % jdx + "\t" + '\t'.join([str(cm[idx][jdx]) for idx in range(args.nlabels)]))

    if args.logit_diff:
        y = np.zeros((total_rows, max_row_by_sentence, 1))
        za = np.zeros((total_rows, max_row_by_sentence, 1))
        z1 = np.zeros((total_rows, max_row_by_sentence, 1))
        z2 = np.zeros((total_rows, max_row_by_sentence, 1))
    else:
        y = np.zeros((total_rows, max_row_by_sentence, args.nlabels))
        za = np.zeros((total_rows, max_row_by_sentence, args.nlabels))
        z1 = np.zeros((total_rows, max_row_by_sentence, args.nlabels))
        z2 = np.zeros((total_rows, max_row_by_sentence, args.nlabels))
    print("Constructing comparative logit matrices of shape " + str(y.shape) + "...")
    ridx = None
    last_row_size = None
    row_to_idxs = {}  # map from infill example rows back to their structure idxs
    idx_to_row = {}
    sample_a_seen = 0
    sample_1_seen = 0
    sample_2_seen = 0
    for idx in range(len(d["inputs"])):
        # on a source sentence, populate expected rows of masked alternatives with true logits
        # they will be overwritten in subsequent passes if they were tested / informative
        if d["maskdata"][idx][1] == -1:
            if ridx is None:
                ridx = 0
            else:
                ridx += last_row_size
            last_row_size = ids_to_nrows[d["maskdata"][idx][0]]
            for jdx in range(ids_to_nrows[d["maskdata"][idx][0]]):
                if args.logit_diff:
                    y[ridx + jdx, jdx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                    za[ridx + jdx, jdx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                    z1[ridx + jdx, jdx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                    z2[ridx + jdx, jdx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                else:
                    y[ridx+jdx, jdx, :] = d["logits"][idx]  # true logits for all comparative masks
                    za[ridx + jdx, jdx, :] = d["logits"][idx]  # will be overwritten if infill argmaxed alternatives
                    z1[ridx + jdx, jdx, :] = d["logits"][idx]  # will be overwritten if infill sampled alternatives
                    z2[ridx + jdx, jdx, :] = d["logits"][idx]  # will be overwritten if infill sampled alternatives
        else:
            # model output logits for this mask because infill gave diff word
            mask_idx = d["maskdata"][idx][1]
            sample_idx = d["maskdata"][idx][2]
            if ridx + mask_idx not in row_to_idxs:
                row_to_idxs[ridx + mask_idx] = [None, None, None]  # argmax, sample 1, sample 2 data source
            if sample_idx == 0:
                if args.logit_diff:
                    za[ridx + mask_idx, mask_idx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                else:
                    za[ridx + mask_idx, mask_idx, :] = d["logits"][idx]
                sample_a_seen += 1
                row_to_idxs[ridx + mask_idx][0] = idx
            elif sample_idx == 1:
                if args.logit_diff:
                    z1[ridx + mask_idx, mask_idx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                else:
                    z1[ridx + mask_idx, mask_idx, :] = d["logits"][idx]
                sample_1_seen += 1
                row_to_idxs[ridx + mask_idx][1] = idx
            elif sample_idx == 2:
                if args.logit_diff:
                    z2[ridx + mask_idx, mask_idx, :] = d["logits"][idx][1] - d["logits"][idx][0]
                else:
                    z2[ridx + mask_idx, mask_idx, :] = d["logits"][idx]
                sample_2_seen += 1
                row_to_idxs[ridx + mask_idx][2] = idx
            else:
                sys.exit("ERROR: unrecognized sample_idx=%d" % sample_idx)
            idx_to_row[idx] = ridx + mask_idx
    print("... done; saw %d samples from argmax, %d samples from draw 0, and %d samples from draw 1"
          % (sample_a_seen, sample_1_seen, sample_2_seen))

    print("Gathering sampling statistics...")
    s1_diff_a = s2_diff_a = s1_diff_s2 = 0
    for ridx in row_to_idxs:
        ridx_a, ridx_1, ridx_2 = row_to_idxs[ridx]
        if ridx_1 is not None:
            if (ridx_a is None or
                    d["inputs"][ridx_a][d["maskdata"][ridx_a][1]] != d["inputs"][ridx_1][d["maskdata"][ridx_a][1]]):
                s1_diff_a += 1
            if (ridx_2 is None or
                    d["inputs"][ridx_2][d["maskdata"][ridx_2][1]] != d["inputs"][ridx_1][d["maskdata"][ridx_2][1]]):
                s1_diff_s2 += 1
        elif ridx_a is not None:
            s1_diff_a += 1  # does not differ from original, since ridx_1 is None
        elif ridx_2 is not None:
            s1_diff_s2 += 1
        if ridx_2 is not None:
            if (ridx_a is None or
                    d["inputs"][ridx_a][d["maskdata"][ridx_a][1]] != d["inputs"][ridx_2][d["maskdata"][ridx_a][1]]):
                s2_diff_a += 1
        elif ridx_a is not None:
            s2_diff_a += 1  # does not differ from original, since ridx_1 is None

    frows = float(num_source * max_row_by_sentence)
    print("... done:\n\tam match orig:\t%.3f\n\ts1 match orig:\t%.3f\n\ts1 match am:\t%.3f" %
          (1 - sample_a_seen / frows, 1 - sample_1_seen / frows, 1 - s1_diff_a / frows) +
          "\n\ts2 match orig:\t%.3f\n\ts2 match am:\t%.3f\n\ts1 match s2:\t%.3f" %
          (1 - sample_2_seen / frows, 1 - s2_diff_a / frows, 1 - s1_diff_s2 / frows))

    # Calculate T.q
    print("Calculating T...")
    if args.test == "directed":
        T = y - z1
        print("... done; used directed calculation T = y - z1")
    elif args.test == "directionless":
        T = (y - z2)**2 - (z1 - z2)**2
        print("... done; used directionless calculation T = (y - z2)**2 - (z1 - z2)**2")
    elif args.test == "directionless-argmax":
        T = (y - za)**2 - (z1 - za)**2
        print("... done; used directionless-argmax calculation T = (y - za)**2 - (z1 - za)**2")

    # Let alpha be the FDR threshold in [0, 1]
    print("Calculating t for alpha=%f" % args.alpha)
    flat_t = T[~(np.isnan(T) | (T == 0))]
    tries = 0
    if len(flat_t) == 0:
        t = np.nan
        print('... No non-zero results.')
    elif flat_t.min() > 0:
        t = flat_t.min()
        print('... All positive results!', len(flat_t))
    else:
        t_options = list(np.sort(flat_t[flat_t < 0])[::-1]) + [flat_t.min() - 1e-6]
        t = np.nan
        for o in t_options:
            cur_alpha = (flat_t <= o).sum() / float(max(1, (flat_t >= -o).sum()))
            if cur_alpha <= args.alpha:
                t = -o
                print('... Success!', t, cur_alpha, (flat_t >= t).sum())
                print("... tried %d / %d t value options" % (tries, len(t_options)))
                break
            else:
                tries += 1

    # Plot the histogram of test statistics.
    plt.hist(flat_t, bins=50)
    if not np.isnan(t):
        plt.axvline(t, ls='--', color='red')
    plt.savefig(os.path.join(args.output_dir, 'knockoff-stats.pdf'), bbox_inches='tight')
    plt.close()
    plt.hist([ft for ft in flat_t if ft < 0], bins=50)
    if not np.isnan(t):
        plt.axvline(t, ls='--', color='red')
    plt.savefig(os.path.join(args.output_dir, 'knockoff-stats-less0.pdf'), bbox_inches='tight')
    plt.close()
    print("... done")

    # Select all elements in T greater than or equal to t. these are discoveries at the alpha FDR level.
    # check whether any one of K classes exceed t.
    print("Looking for discoveries at the alpha FDR level...")
    t_logit_size = 1 if args.logit_diff else args.nlabels
    rows_of_interest = []
    num_logits_above_t = 0
    for ridx in range(total_rows):  # TODO: there is a much more efficient loop than this based on orig structure
        for widx in range(max_row_by_sentence):
            logits_above_threshold = [T[ridx, widx, kidx] >= t for kidx in range(t_logit_size)]
            if True in logits_above_threshold:
                rows_of_interest.append([ridx, logits_above_threshold, [T[ridx, widx, kidx]
                                                                        for kidx in range(t_logit_size)]])
            num_logits_above_t += logits_above_threshold.count(True)
    print("... done; found %d / %d (%.2f) rows of interest with %d / %d (%.2f) logits above threshold" %
          (len(rows_of_interest), total_rows, len(rows_of_interest) / float(total_rows),
           num_logits_above_t, total_rows * t_logit_size, num_logits_above_t / float(total_rows * t_logit_size)))

    # number of true positive and false positive sentence labels from the model for deeper analysis.
    ids_to_logits = {sid: d["logits"][ids_to_idxs[sid]] for sid in ids_to_idxs}
    ids_to_probs = {sid: softmax(ids_to_logits[sid]) for sid in ids_to_idxs}
    for lidx in range(args.nlabels):
        ids_to_sel_probs = {sid: ids_to_probs[sid][lidx] for sid in ids_to_idxs if ids_to_probs[sid][lidx] > 0.5 and
                            ((args.label_match == 0 and d["labels"][ids_to_idxs[sid]] != lidx) or
                             (args.label_match == 1 and d["labels"][ids_to_idxs[sid]] == lidx) or
                             args.label_match == -1)}
        sel_ids = [(k, ids_to_sel_probs[k])
                   for k in sorted(ids_to_sel_probs, key=ids_to_sel_probs.get, reverse=True)]
        if args.random_show:
            random.shuffle(sel_ids)
            print("Random samples for decision label=%d:" % lidx)
        else:
            print("Highest model confidence samples for decision label=%d:" % lidx)
        sel_ids = sel_ids[:args.m]
        for sid, prob in sel_ids:
            tks = d["inputs"][ids_to_idxs[sid]]
            print("\tOriginal sentence (conf=%.5f, gl=%d" % (prob, d["labels"][ids_to_idxs[sid]]) + "): " +
                  sentence_str(tks))
            idx = ids_to_idxs[sid] + 1  # start with data idx after source sentence
            sid_above = [False for _ in range(len(d["inputs"][ids_to_idxs[sid]]))]
            sid_lat = [[None for _ in range(t_logit_size)] for _ in range(len(d["inputs"][ids_to_idxs[sid]]))]
            sid_logits = [None for _ in range(len(d["inputs"][ids_to_idxs[sid]]))]
            while d["maskdata"][idx][0] == sid:  # while this word relates to the source sentence
                for ridx, lat, rl in rows_of_interest:
                    if idx in row_to_idxs[ridx]:  # this sample contributed to a row of interest above the FDR threshold
                        jdx = d["maskdata"][idx][1]
                        sid_above[jdx] = True
                        sid_lat[jdx] = lat
                        sid_logits[jdx] = rl
                        # print(row_to_idxs[ridx])  # DEBUG
                        # DEBUG - view argmax and sample alternatives
                        # DEBUG - view argmax and sample alternatives
                        # print("%s -> (%s, %s, %s)" %
                        #       (d["inputs"][ids_to_idxs[sid]][jdx],
                        #        d["inputs"][row_to_idxs[ridx][0]][jdx] if row_to_idxs[ridx][0] is not None else None,
                        #        d["inputs"][row_to_idxs[ridx][1]][jdx] if row_to_idxs[ridx][1] is not None else None,
                        #        d["inputs"][row_to_idxs[ridx][2]][jdx] if row_to_idxs[ridx][2] is not None else None))
                idx += 1
            # Print each word and its thresholds / logits
            for jdx in range(len(sid_above)):
                if sid_above[jdx]:
                    print("\t\t%s\t" % d["inputs"][ids_to_idxs[sid]][jdx].encode('ascii', 'ignore').decode('ascii')
                          + str(sid_lat[jdx]) +
                          "\t" + str(sid_logits[jdx]))

            # Pyplot the text in color for words above threshold indicating which logit shifted when the word was
            # replaced with samples.
            ls = sentence_str(tks).split()
            # chop off [PAD] tokens (done by sentence_str conversion) and create colors arrays
            slc = sid_lat[:len(ls)]
            # Fill colors from logit changes above threshold.
            if args.nlabels == 2:
                if args.logit_diff:
                    lc = [[0, 0, 1 if l0[0] else 0] for l0 in slc]
                else:
                    lc = [[1 if l0 else 0, 0, 1 if l1 else 0] for l0, l1 in slc]
            elif args.nlabels == 3:
                lc = [[0 if l0 else 0.5, 0 if l1 else 0.5, 0 if l2 else 0.5] for l0, l1, l2 in slc]
            else:
                sys.exit("ERROR: haven't word visualization for nlabels=%d" % args.nlabels)
            plt.figure()
            t = plt.gca().transData
            fig = plt.gcf()
            break_every_n_words = 6
            xspace = 1. / break_every_n_words
            yspace = 0.04
            wc = zip(ls, lc)
            latex_text = []
            for idx in range(len(wc)):
                s, c = wc[idx]
                col = idx % break_every_n_words
                line = idx // break_every_n_words
                text = plt.text(xspace + xspace * col, 1 - yspace - yspace*line,
                                " " + s + " ", color=c, transform=t, size=10)
                text.draw(fig.canvas.get_renderer())

                if args.nlabels == 2:
                    if args.logit_diff:
                        if slc[idx][0]:
                            # latex_text.append("\lboth{" + s + "}(%.3f)" % sid_logits[idx][0])  # with T value
                            latex_text.append("\lboth{" + s + "}")  # without T value
                        else:
                            latex_text.append(s)
                    else:
                        if slc[idx][0] and slc[idx][1]:
                            latex_text.append("\lboth{" + s + "}")
                        elif slc[idx][0]:
                            latex_text.append("\lneg{" + s + "}")
                        elif slc[idx][1]:
                            latex_text.append("\lpos{" + s + "}")
                        else:
                            latex_text.append(s)
            plt.axis('off')
            plt_name = "l-%d_p-%d_id-%d.pdf" % (d["labels"][ids_to_idxs[sid]], lidx, sid)
            plt.savefig(os.path.join(args.output_dir, plt_name), bbox_inches='tight')
            plt.close()

            # merge BPEs in latex text for readability
            idx = 0
            while idx < len(latex_text) - 1:
                if '#' in latex_text[idx + 1]:  # BPE is followed by a suffix; merge them
                    latex_text[idx] = latex_text[idx] + latex_text[idx + 1].replace('#', '')
                    del latex_text[idx + 1]
                else:
                    idx += 1
            print("l-%d, p-%d, text: %s" % (d["labels"][ids_to_idxs[sid]], lidx, ' '.join(latex_text)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', type=str, required=True,
                        help="json input file produced by run_classifier.py mask_eval")
    parser.add_argument('--nlabels', type=int, required=True,
                        help="number of class labels")
    parser.add_argument('--test', type=str, required=True,
                        help="either 'directed', 'directionless', or 'directionless-argmax'")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="where to dump histogram and example sentences")
    parser.add_argument('--logit_diff', type=int, required=False, default=1,
                        help="if 0, use distinct classes")
    parser.add_argument('--alpha', type=float, required=False, default=0.2,
                        help="the alpha level")
    parser.add_argument('--m', type=int, required=False, default=3,
                        help="the number of high confidence samples to show per class")
    parser.add_argument('--label_match', type=int, required=False, default=-1,
                        help=("if 0, require that predicted and gold labels do not match in shown samples; " +
                              "if 1, require that they do"))
    parser.add_argument('--random_show', type=int, required=False, default=-1,
                        help="if 1, show random instead of high confidence")
    main(parser.parse_args())
