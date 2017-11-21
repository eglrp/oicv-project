# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
from __future__ import print_function

import os

import errno
import numpy as np
import time


import logging
from pandas import ExcelFile

import matplotlib
matplotlib.use('Agg')

from pystruct.models import ChainCRF  # noqa
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM  # noqa
from sklearn.cross_validation import KFold  # noqa
from sklearn.model_selection import ParameterSampler  # noqa
from sklearn.svm import LinearSVC  # noqa

from plot_segments import plot_segments  # noqa

import matplotlib.pyplot as plt  # noqa

""" CONSTANTS """

structured_learning_options = {
    'oneslack': OneSlackSSVM,
    'nslack': NSlackSSVM,
    'frankwolfe': FrankWolfeSSVM,
}
num_segments_per_jacket = 40
num_segments_classes = 12
feature_sets_options = {'basic', 'basic_and_middle', 'full'}
features_names = [
    'x0_norm',
    'y0_norm',
    'x1_norm',
    'y1_norm',
    'angle',
    'x_norm_middle',
    'y_norm_middle',
    'x0',
    'y0',
    'x1',
    'y1',
    'distance',
]
name_of_labels = [
    'neck',
    'left shoulder',
    'outer left sleeve',
    'left wrist',
    'inner left sleeve',
    'left chest',
    'waist',
    'right chest',
    'inner right sleeve',
    'right wrist',
    'outer right sleeve',
    'right shoulder',
]


""" FUNCTIONS """


def setup_logging(log_path):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(name)s:%(lineno)d %(levelname)s :: %(message)s')

    # Create file handler, attach formatter and add it to the logger
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Write to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def create_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_segments_data():
    path_measures = 'man_jacket_hand_measures.xls'
    xl = ExcelFile(path_measures)
    sheet = xl.parse(xl.sheet_names[0])
    """ be careful, parse() just reads literals, does not execute formulas """
    xl.close()
    it = sheet.iterrows()
    labels_segments = []
    segments = []
    for row in it:
        ide = row[1]['ide']
        segments.append(np.load(os.path.join('segments', ide + '_front.npy')))
        labels_segments.append(list(row[1].values[-num_segments_per_jacket:]))
    labels_segments = np.array(labels_segments).astype(int)
    return labels_segments, segments, sheet


def select_features_matrix(feature_set, num_jackets, segments):
    if feature_set == 'basic':
        num_features = 5
    elif feature_set == 'basic_and_middle':
        num_features = 7
    elif feature_set == 'full':
        num_features = 12
    else:
        raise ValueError(
            "Feature set option {!r} not available. Choose one of: 'basic', 'basic_and_middle', 'full'".format(
                feature_set)
        )
    X = np.zeros((num_jackets, num_segments_per_jacket, num_features))
    for jacket_segments, i in zip(segments, range(num_jackets)):
        for s, j in zip(jacket_segments, range(num_segments_per_jacket)):
            """ set the features """
            if feature_set == 'basic':
                features = s.x0norm, s.y0norm, s.x1norm, s.y1norm, s.angle
            elif feature_set == 'basic_and_middle':
                features = s.x0norm, s.y0norm, s.x1norm, s.y1norm, s.angle / (2 * np.pi), \
                           (s.x0norm + s.x1norm) / 2., (s.y0norm + s.y1norm) / 2.
            elif feature_set == 'full':
                features = s.x0norm, s.y0norm, s.x1norm, s.y1norm, s.angle, \
                           (s.x0norm + s.x1norm) / 2., (s.y0norm + s.y1norm) / 2., \
                           s.x0, s.y0, s.x1, s.y1, np.sqrt(
                    (s.x0norm - s.x1norm) ** 2 + (s.y0norm - s.y1norm) ** 2)
            X[i, j, 0:num_features] = features
    return X, num_features


def train_predict_and_score(model, X_train, Y_train, X_test, Y_test):
    start = time.time()
    model.fit(X_train, Y_train)
    training_time = time.time() - start

    Y_pred = np.array(model.predict(X_test))

    score = model.score(X_test, Y_test)

    return training_time, Y_pred, score


def report_global_results(num_jackets, scores_crf, scores_svm, training_time_crf, training_time_svm,
                          wrong_segments_crf, wrong_segments_svm):
    logger = logging.getLogger('structured_prediction.report_measures')
    total_segments = num_jackets * num_segments_per_jacket
    wrong_segments_crf = np.array(wrong_segments_crf)
    wrong_segments_svm = np.array(wrong_segments_svm)
    logger.info('Results per fold ')
    logger.info('Scores CRF : {}'.format(scores_crf))
    logger.info('Scores SVM : {}'.format(scores_svm))
    logger.info('Wrongs CRF : {}'.format(wrong_segments_crf))
    logger.info('Wrongs SVM : {}'.format(wrong_segments_svm))
    logger.info('Training time CRF (s): {}'.format(training_time_crf))
    logger.info('Training time SVM (s): {}'.format(training_time_svm))
    logger.info(' ')
    final_score_crf = 1.0 - wrong_segments_crf.sum() / float(total_segments)
    logger.info('Final score CRF: {}, {} wrong labels in total out of {}'.format(
        final_score_crf,
        wrong_segments_crf.sum(),
        total_segments
    ))
    final_score_svm = 1.0 - wrong_segments_svm.sum() / float(total_segments)
    logger.info('Final score SVM: {}, {} wrong labels in total out of {}'.format(
        final_score_svm,
        wrong_segments_svm.sum(),
        total_segments)
    )
    logger.info('Average training time CRF: {:.3f} s'.format(np.mean(training_time_crf)))
    logger.info('Average training time SVM: {:.3f} s'.format(np.mean(training_time_svm)))

    return final_score_crf, final_score_svm


def plot_unary_weights(C, feature_set, num_features, ssvm, save_path):
    unary_weights = ssvm.w[:num_segments_classes * num_features].reshape(num_segments_classes, num_features)
    plt.figure(figsize=(10, 10))
    plt.imshow(unary_weights)
    plt.colorbar()
    plt.title("Transition parameters of the chain CRF.")
    plt.xticks(np.arange(num_features), features_names[:num_features], rotation=45)
    plt.yticks(np.arange(num_segments_classes), name_of_labels)
    result_name = '{C:.2f}_{feature_set}_unarry_coefficient.png'.format(C=C, feature_set=feature_set)
    plt.savefig(os.path.join(save_path, result_name))
    plt.close()


def plot_pairwise_weights(C, feature_set, num_features, ssvm, save_path):
    pairwise_weights = ssvm.w[num_segments_classes * num_features:].reshape(
        num_segments_classes, num_segments_classes
    )
    plt.figure(figsize=(12, 10))
    plt.imshow(pairwise_weights)
    plt.colorbar()
    plt.title("Transition parameters of the chain CRF.")
    plt.xticks(np.arange(num_segments_classes), name_of_labels, rotation=45)
    plt.yticks(np.arange(num_segments_classes), name_of_labels)
    result_name = '{C:.2f}_{feature_set}_pairwise_coefficient.png'.format(C=C, feature_set=feature_set)
    plt.savefig(os.path.join(save_path, result_name))
    plt.close()


def main():
    """ Main function """

    """ PARAMETERS """

    random_search_hyperparams = False
    random_search_iters = 100

    C = 89.3
    feature_set_name = 'basic_and_middle'  # One of: 'basic', 'basic_and_middle' and 'full'

    add_gaussian_noise_to_features = False
    sigma_noise = 0.1

    plot_example = False
    plot_example_number = 4
    plot_labeling = True
    plot_coefficients = True

    structured_learning_algorithm = 'nslack'  # One of: 'oneslack', 'nslack', 'frankwolfe'

    n_folds = 5

    """ MAIN """

    save_path = os.path.abspath('results')
    predictions_path = os.path.abspath('predictions')
    create_dirs(save_path)
    create_dirs(predictions_path)

    setup_logging(os.path.join(save_path, 'structured_prediction.log'))
    logger = logging.getLogger('structured_prediction.main')

    # Load the segments and the groundtruth for all jackets
    Y, segments, sheet = load_segments_data()
    num_jackets = Y.shape[0]

    # Show groundtruth for 3rd jacket
    if plot_example:
        plot_segments(
            segments[plot_example_number],
            sheet.ide[plot_example_number],
            Y[plot_example_number]
        )

    if random_search_hyperparams:
        # Random search
        random_search_space = {
            'feature_set_name': list(feature_sets_options),
            'C': np.logspace(0, 2.7, 1000000)
        }
        sample_generator = ParameterSampler(param_distributions=random_search_space, n_iter=random_search_iters)
    else:
        # Fixed values
        sample_generator = [{'feature_set_name': feature_set_name, 'C': C}]

    objective_metric = 0
    best_model_params = dict()
    for d in sample_generator:

        feature_set_name = d['feature_set_name']
        C = d['C']

        logger.info("\n")
        logger.info('-----------------------------------------')
        logger.info('Feature Set: {}'.format(feature_set_name))
        logger.info('C: {:.4f}'.format(C))
        logger.info('Learner: {}'.format(structured_learning_algorithm.upper()))
        logger.info('-----------------------------------------')

        X, num_features = select_features_matrix(feature_set_name, num_jackets, segments)

        # Add some noise to the features (optional)
        if add_gaussian_noise_to_features:
            logger.info('Noise sigma {}'.format(sigma_noise))
            X = X + np.random.normal(0.0, sigma_noise, size=X.size).reshape(np.shape(X))

        # DEFINE HERE YOUR GRAPHICAL MODEL AND CHOOSE ONE LEARNING METHOD
        # (OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM)
        model = ChainCRF()
        try:
            ssvm_class = structured_learning_options[structured_learning_algorithm]
        except KeyError:
            raise Exception('{!r} structured learning model not supported. Use one of: {!r}'.format(
                structured_learning_algorithm,
                structured_learning_options.keys(),
            ))
        ssvm = ssvm_class(model=model, C=C)

        # LINEAR SVM
        svm = LinearSVC(C=C)

        # with 5 in each fold we have 4 jackets for testing, 19 for training,
        # with 23 we have leave one out : 22 for training, 1 for testing
        scores_crf = np.zeros(n_folds)
        scores_svm = np.zeros(n_folds)
        training_time_crf = np.zeros(n_folds)
        training_time_svm = np.zeros(n_folds)
        wrong_segments_crf = []
        wrong_segments_svm = []

        kf = KFold(num_jackets, n_folds=n_folds)
        for fold, (train_index, test_index) in enumerate(kf):
            logger.info(' ')
            logger.info('train index: {}'.format(train_index))
            logger.info('test index: {}'.format(test_index))
            logger.info('{} jackets for training, {} for testing'.format(len(train_index), len(test_index)))
            X_train = X[train_index]
            Y_train = Y[train_index]
            X_test = X[test_index]
            Y_test = Y[test_index]

            # YOUR S-SVM TRAINING CODE HERE
            crf_training_time, Y_pred, crf_score = train_predict_and_score(
                ssvm, X_train, Y_train, X_test, Y_test
            )
            logger.info('CRF training time: {:.2f} s'.format(crf_training_time))
            training_time_crf[fold] = crf_training_time
            logger.info("Test score with chain CRF: {}".format(crf_score))
            scores_crf[fold] = crf_score
            wrong_segments_array = Y_pred != Y_test
            wrong_segments_crf.append(np.count_nonzero(wrong_segments_array))

            # figure showing the result of classification of segments for
            # each jacket in the testing part of present fold
            if plot_labeling:
                for ti, pred in zip(test_index, Y_pred):
                    s = segments[ti]
                    prediction_filepath = os.path.join(predictions_path, 'crf_jacket_{}-C_{}-features_{}.png'.format(
                        ti + 1, C, feature_set_name
                    ))
                    plot_segments(s, caption='CRF predictions for jacket {}'.format(ti + 1), labels_segments=pred,
                                  savepath=prediction_filepath)

            # YOUR LINEAR SVM TRAINING CODE HERE """
            X_train_svm, X_test_svm, Y_train_svm, Y_test_svm = np.vstack(X_train), np.vstack(X_test), \
                                                               np.hstack(Y_train), np.hstack(Y_test)
            svm_training_time, Y_pred_svm, svm_score = train_predict_and_score(
                svm, X_train_svm, Y_train_svm, X_test_svm, Y_test_svm
            )

            logger.info('LinearSVM training time: {:.2f} s'.format(svm_training_time))
            training_time_svm[fold] = svm_training_time
            logger.info("Test score with linear SVM: {}".format(svm_score))
            scores_svm[fold] = svm_score
            wrong_segments_array = Y_pred_svm != Y_test_svm
            wrong_segments_svm.append(np.count_nonzero(wrong_segments_array))

            # figure showing the result of classification of segments for
            # each jacket in the testing part of present fold
            Y_pred_svm = np.reshape(Y_pred_svm, (-1, num_segments_per_jacket))
            if plot_labeling:
                for ti, pred in zip(test_index, Y_pred_svm):
                    s = segments[ti]
                    prediction_filepath = os.path.join(predictions_path, 'svm_jacket_{}-C_{}-features_{}.png'.format(
                        ti + 1, C, feature_set_name
                    ))
                    plot_segments(s, caption='LinearSVM predictions for jacket {}'.format(ti + 1), labels_segments=pred,
                                  savepath=prediction_filepath)

        # Global results
        final_score_crf, _ = report_global_results(
            num_jackets, scores_crf, scores_svm, training_time_crf, training_time_svm,
            wrong_segments_crf, wrong_segments_svm
        )

        if random_search_hyperparams and final_score_crf > objective_metric:
            objective_metric = final_score_crf
            best_model_params = d
            logger.info('Found best model so far with parameters: {!r}'.format(d))
            logger.info('Found best model so far with objective score: {!r}'.format(objective_metric))

        if plot_coefficients:
            plot_unary_weights(C, feature_set_name, num_features, ssvm, save_path)
            plot_pairwise_weights(C, feature_set_name, num_features, ssvm, save_path)

        logger.info("End of sample parameters")

    if random_search_hyperparams:
        logger.info('Best model objective metric: {!r}'.format(objective_metric))
        logger.info('Best model parameters: {!r}'.format(best_model_params))


if __name__ == '__main__':
    main()
