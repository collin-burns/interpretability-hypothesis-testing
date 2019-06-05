#!/usr/bin/env python3
# Takes the directory where the raw LMRD lives and converts it to the .tsv train and test formats expected by
# the huggingface BERT run_classifier.py script.

import argparse
import random
import os
import spacy
from tqdm import tqdm


def main(args):
    nlp = spacy.load('en')

    print("converting LMRD...")
    for cond in ["test", "train"]:
        datalines = []  # 'label[TAB]review'
        for li, ls in [[1, "pos"], [0, "neg"]]:
            d = os.path.join(args.input_dir, cond, ls)
            c = 0
            for root, _, fs in os.walk(d):
                for fn in fs:
                    if fn.split('.')[1] == 'txt':  # a review file
                        with open(os.path.join(root, fn), 'r') as f:
                            review = f.read().replace('<br />', ' ').strip()
                        datalines.append("%d\t%s" % (li, review))
                        c += 1
            print("... read %d reviews from cond %s, label %s" % (c, cond, ls))

        of = os.path.join(args.output_dir, cond + ".tsv")
        print("... shuffling %d read LMRD %s datalines and writing to %s" % (len(datalines), cond, of))
        random.shuffle(datalines)
        if args.test_output_limit is not None and cond == "test":
            datalines = datalines[:args.test_output_limit]
            print("...... trimmed output data to size %d" % len(datalines))
        if args.train_output_limit is not None and cond == "train":
            datalines = datalines[:args.train_output_limit]
            print("...... trimmed output data to size %d" % len(datalines))
        with open(of, 'w') as f:
            f.write('\n'.join(datalines))
        print("...... done")

        print("... splitting %d LMRD datalines into sentences..." % len(datalines))
        sentences = []  # list of sentences per review
        for review in tqdm(datalines):
            doc = nlp(review.split('\t')[1])
            sentences.append([sent.string.strip() for sent in doc.sents])
        of = os.path.join(args.output_dir, cond + ".lang.txt")
        print("... done; split %d dataline documents into %d sentences; writing to %s" %
              (len(sentences), sum([len(sentences[idx]) for idx in range(len(sentences))]), of))
        with open(of, 'w') as f:
            for idx in range(len(sentences)):
                f.write('\n'.join(sentences[idx]) + '\n\n')
        print("...... done")
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="where the LMRD lives")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="where to dump train.tsv, test.tsv, train.lang.txt, test.lang.txt")
    parser.add_argument('--train_output_limit', type=int, required=False,
                        help="set artificial cap on the size of the train set")
    parser.add_argument('--test_output_limit', type=int, required=False,
                        help="set artificial cap on the size of the test set")
    main(parser.parse_args())
