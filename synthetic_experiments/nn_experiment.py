from __future__ import print_function
import argparse
import shap
import lime.lime_tabular as tabular
import numpy as np
import torch
import tensorflow as tf
import os
import keras
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import random
from deepexplain.tensorflow import DeepExplain
BATCH_SIZE = 1000

def get_prediction(X):
    Y = np.sum(np.abs(X), axis=1)

    return np.expand_dims(Y, axis=1)

def get_feature(lime_string):
    """
    Manual parsing of the string output by lime to get the feature corresponding to the given feature value.
    """
    num_ineqs = 0
    for l in lime_string:
        if l == "<" or l  == ">":
            num_ineqs += 1

    feature_str = ""
    if num_ineqs == 1:  # then the feature name is the first thing, followed by a space
        for l in lime_string:
            if l != " ":
                feature_str += l
            else:
                return int(feature_str)
    else:  # necessarily equal to 2 then
        # start recording after the second space, and stop before the third space
        nspaces = 0
        recording = False
        for l in lime_string:
            if l == " ":
                nspaces += 1
                if nspaces == 2:
                    recording = True
                elif nspaces == 3:
                    return int(feature_str)
            elif recording == True:
                feature_str += l

def get_output_fn(model):
    """
    Input for shap.KernelExplainer() and for LIME
    User supplied function that takes a matrix of samples (# samples x # features) and computes a the output of the
    model for those samples. The output can be a vector (# samples) or a matrix (# samples x # model outputs).
    """
    def get_output(X):
        y = model.predict(X)
        return np.squeeze(y)
    return get_output

def create_rank(scores):
    """
    Compute rank of each feature based on weight.
    Adapted from https://github.com/Jianbo-Lab/L2X/tree/master/synthetic
    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank=(-permutated_weights).argsort().argsort()+1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)

    return np.array(ranks)

class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.
    From https://github.com/Jianbo-Lab/L2X/tree/master/synthetic
    """
    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random_uniform(shape =(batch_size, self.k, d),
                                    minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval = 1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape

def L2X(X, y, k, nepochs=10, tau=0.1):
    """
    Assumes y is the output of the model (a real number).
    Adapted from https://github.com/Jianbo-Lab/L2X/tree/master/synthetic
    """
    nfeatures = X.shape[1]
    input_shape = nfeatures

    activation = "relu"
    # P(S|X)
    model_input = Input(shape=(input_shape,), dtype="float32")
    net = Dense(100, activation=activation, name = "s/dense1",
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = Dense(100, activation=activation, name = "s/dense2",
                kernel_regularizer=regularizers.l2(1e-3))(net)

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = Dense(input_shape)(net)
    samples = Sample_Concrete(tau, k, name = "sample")(logits)

    # q(X_S)
    new_model_input = Multiply()([model_input, samples])
    net = Dense(200, activation=activation, name = "dense1",
                kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
    net = BatchNormalization()(net) # Add batchnorm for stability.
    net = Dense(200, activation=activation, name = "dense2",
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)
    preds = Dense(1, activation="linear", name = "dense4",
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)
    adam = optimizers.Adam(lr = 1e-3)
    model.compile(loss="mean_squared_error",  # changed from original classification task
                  optimizer=adam)

    model.fit(X, y, epochs=nepochs, batch_size=BATCH_SIZE, verbose=0)
    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None,
                       optimizer="rmsprop",
                       metrics=[None])
    scores = pred_model.predict(X, verbose = 0, batch_size = BATCH_SIZE)
    ranks = create_rank(scores)
    return scores, ranks

def get_independent_data(nsamples, nfeatures, interesting_mean, interesting_prob):
    Z = np.random.uniform(size=(nsamples, nfeatures)) < interesting_prob
    X = np.random.normal(size=(nsamples, nfeatures))
    X_interesting = np.random.normal(interesting_mean, size=(nsamples, nfeatures))
    X[Z] = X_interesting[Z]
    Y = get_prediction(X)

    return X, Y, Z

def sample(x, beta, j):
    if j == 0:
        return np.random.normal()
    else:
        mean = beta[:j-1].dot(x[:j-1])
        return np.random.normal(loc=mean)

def replace_features(X_fake, j, beta, independent=True):
    if independent:
        X_fake[:, j] = np.random.normal(size=X_fake.shape[0])
    else:
        for i in range(X_fake.shape[0]):
            X_fake[i, j] = sample(X_fake[i], beta, j)

    return X_fake

def get_correlated_data(nsamples, nfeatures, interesting_mean, interesting_prob, beta):
    Z = np.random.uniform(size=(nsamples, nfeatures)) < interesting_prob
    X = np.random.normal(loc=interesting_mean, size=(nsamples, nfeatures))

    for i in range(nsamples):
        for j in range(nfeatures):
            if not Z[i, j]: # non-interesting feature
                X[i, j] = sample(X[i], beta, j)
    Y = get_prediction(X)

    return X, Y, Z

def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ind = "independent" if args.independent else "correlated"
    experiment_dir = args.experiment_dir.format(ind, args.experiment_num)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # save some experiment information in the experiment directory for reference
    with open(os.path.join(experiment_dir, "experiment_info.txt"), "w") as f:
        f.write("seed: {}, ntrain: {}, ntest: {}, nfeatures: {}, interesting_prob: {}, interesting_mean: {},"
                "nperms: {}, nruns: {}, nepochs: {}, experiment_num: {}, skip_l2x: {}, skip_lime: {} "
                "threshold: {}, ntrain_epochs: {}, hidden_size: {}, independent: {}".format(
                args.seed, args.ntrain, args.ntest, args.nfeatures, args.interesting_prob, args.interesting_mean,
                args.nperms, args.nruns, args.nepochs, args.experiment_num, args.skip_l2x, args.skip_lime,
                args.threshold, args.ntrain_epochs, args.hidden_size, args.independent))

    for run in range(args.nruns):
        K.clear_session()
        print("""""""""""""""""")
        print("Starting run {}/{}".format(run+1, args.nruns))
        print("""""""""""""""""")

        # get data for training neural net
        beta = (1/4)*np.random.normal(size=args.nfeatures)
        if args.independent:
            X, Y, important = get_independent_data(args.ntrain, args.nfeatures, args.interesting_mean, args.interesting_prob)
            X_test, Y_test, important_test = get_independent_data(args.ntest, args.nfeatures, args.interesting_mean, args.interesting_prob)
            X_val, Y_val, important_val = get_independent_data(1000, args.nfeatures, args.interesting_mean, args.interesting_prob)
        else:
            X, Y, important = get_correlated_data(args.ntrain, args.nfeatures, args.interesting_mean, args.interesting_prob, beta)
            X_test, Y_test, important_test = get_correlated_data(args.ntest, args.nfeatures, args.interesting_mean, args.interesting_prob, beta)
            X_val, Y_val, important_val = get_correlated_data(1000, args.nfeatures, args.interesting_mean, args.interesting_prob, beta)

        np.save(os.path.join(experiment_dir, "important{}".format(run)), important_test)

        # Train network
        model = Sequential()
        model.add(Dense(args.hidden_size, activation="relu", input_dim=args.nfeatures))
        model.add(Dense(args.hidden_size, activation="relu", input_dim=args.hidden_size))
        model.add(Dense(1, input_dim=args.hidden_size))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        print("avg Y", Y.mean())

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
        model.fit(X, Y, validation_data=(X_val, Y_val), epochs=args.ntrain_epochs, batch_size=100, callbacks=[early_stopping])
        get_output = get_output_fn(model)

        # Evaluate network
        score = model.evaluate(X_test, Y_test)
        print("Test score", score)

        # Get the predictions under the real data and save the important features for each sample
        w = model.predict(X_test)

        # DeepExplain
        with DeepExplain(session=K.get_session()) as de:
            input_tensor = model.layers[0].input
            fModel = Model(inputs=input_tensor, outputs=model.layers[-1].output)
            target_tensor = fModel(input_tensor)
            xs = X_test
            ys = Y_test

            print("Starting grad*input")
            attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
            np.save(os.path.join(experiment_dir, "gradinput{}".format(run)), attributions_gradin)
            print("Starting saliency")
            attributions_sal = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
            np.save(os.path.join(experiment_dir, "saliency{}".format(run)), attributions_sal)
            print("Starting deeplift")
            attributions_dl = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
            np.save(os.path.join(experiment_dir, "deeplift{}".format(run)), attributions_dl)

        # Get and save l2x ranks (except for k=0, which is separately handled).
        # Note that scores are always either 0 (unselected) or 1 (selected).
        if not args.skip_l2x:
            print("Starting L2X")
            all_scores = []
            for i in range(args.nfeatures):
                k = i + 1
                print("L2X k = {}/{}".format(k, args.nfeatures))
                scores, _ = L2X(X_test, w, k, nepochs=args.nepochs)
                all_scores.append(scores)
            np.save(os.path.join(experiment_dir, "l2x_scores{}".format(run)), all_scores)

        # Get and save shapley stats.
        print("Starting SHAP")
        background = X[np.random.choice(X.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(X_test)
        np.save(os.path.join(experiment_dir, "shap_vals{}".format(run)), shap_values)

        if not args.skip_lime:
            # Get and save lime explanations.
            print("Starting LIME")
            explainer = tabular.LimeTabularExplainer(X_test, mode="regression", verbose=False)
            lime_values = np.zeros(shape=(args.ntest, args.nfeatures))
            for i in range(args.ntest):
                explanations = explainer.explain_instance(X[i], get_output, num_features=args.nfeatures).as_list()
                for expl in explanations:
                    feature = get_feature(expl[0])
                    val = expl[1]
                    lime_values[i, feature] = val
            # save
            np.save(os.path.join(experiment_dir, "lime_vals{}".format(run)), np.array(lime_values))

        print("Starting OSFT and IRT")
        # OSFT
        directionless_tstats = np.zeros(X_test.shape)
        directioned_tstats = np.zeros(X_test.shape)

        # IRT
        directionless_pvals = np.zeros(X_test.shape)
        directioned_pvals = np.zeros(X_test.shape)

        # OSFT
        for j in range(args.nfeatures):
            X_fake = np.copy(X_test)
            X_fake = replace_features(X_fake, j, beta, args.independent)
            fake_y = model.predict(X_fake)

            # Directionless: use a null sample to center the predictions and estimate variance
            X_fake = replace_features(X_fake, j, beta, args.independent)
            fake_center = model.predict(X_fake)

            # directionless
            directionless_tstats[:, j] = np.squeeze((w - fake_center) ** 2 - (fake_y - fake_center) ** 2)

            # Directioned
            directioned_tstats[:, j] = np.squeeze(w - fake_y)

            # IRT
            for trial in range(args.nperms):
                X_fake = replace_features(X_fake, j, beta, args.independent)
                fake_y = model.predict(X_fake)

                # directionless
                directionless_pvals[:, j] += np.squeeze((w - fake_center) ** 2 <= (fake_y - fake_center) ** 2)

                # directioned
                directioned_pvals[:, j] += np.squeeze((w <= fake_y))

        # now normalize all the IRT p-values
        directioned_pvals = (1/(args.nperms+1)) * (1 + directioned_pvals)
        directionless_pvals = (1/(args.nperms+1)) * (1 + directionless_pvals)

        # save OSFT and IRT stats
        np.save(os.path.join(experiment_dir, "directioned_tstats{}".format(run)), directioned_tstats)
        np.save(os.path.join(experiment_dir, "directioned_pvals{}".format(run)), directioned_pvals)
        np.save(os.path.join(experiment_dir, "directionless_tstats{}".format(run)), directionless_tstats)
        np.save(os.path.join(experiment_dir, "directionless_pvals{}".format(run)), directionless_pvals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-a", type=int, default=1000000)
    parser.add_argument("--ntrain_epochs", "-o", type=int, default=100)  # for neural net
    parser.add_argument("--ntest", "-n", type=int, default=100)
    parser.add_argument("--nfeatures", "-k", type=int, default=25)
    parser.add_argument("--nperms", "-p", type=int, default=100)  # for IRT
    parser.add_argument("--nruns", "-r", type=int, default=10)  # times to repeat (with new data, etc.)
    parser.add_argument("--hidden_size", "-f", type=int, default=100)  # size of hidden layers
    parser.add_argument("--experiment_num", "-e", type=int, default=0)  # for saving
    parser.add_argument("--experiment_dir", "-x", type=str, default="{}_nn_experiment{}")
    parser.add_argument("--nepochs", "-t", type=int, default=100)  # for L2X
    parser.add_argument("--skip_l2x", "-s", action="store_true")  # for efficiency/quick testing, since L2X is slow.
    parser.add_argument("--skip_lime", "-l", action="store_true")  # for efficiency/quick testing, since LIME is slow.
    parser.add_argument("--seed", "-z", type=int, default=0)
    parser.add_argument("--threshold", "-d", type=float, default=3.0)
    parser.add_argument("--interesting_prob", "-b", type=float, default=0.3)
    parser.add_argument("--interesting_mean", "-m", type=float, default=4)
    parser.add_argument("--independent", "-i", action="store_true")
    args = parser.parse_args()
    main(args)
