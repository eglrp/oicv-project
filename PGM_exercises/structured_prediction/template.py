# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import time

from pandas import ExcelFile

from pystruct.models import ChainCRF, MultiClassClf
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC

from plot_segments import plot_segments

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

""" PARAMETERS """
C = 100
add_gaussian_noise_to_features = False
sigma_noise = 0.1
plot_example = False
plot_labeling = False
plot_coefficients = True
structured_learning_algorithm = 'frankwolfe'  # One of: 'oneslack', 'nslack', 'frankwolfe'
feature_set = 'basic'  # One of: 'basic', 'basic_and_middle' and 'full'

""" 
Load the segments and the groundtruth for all jackets
"""
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

"""
Show groundtruth for 3rd jacket
"""
if plot_example:
    n = 2
    plot_segments(segments[n], sheet.ide[n], labels_segments[n])

""" 
Make matrices X of shape (number of jackets, number of features) 
and Y of shape (number of jackets, number of segments) where, 
for all jackets,
    X = select the features for each segment 
    Y = the grountruth label for each segment
"""
Y = labels_segments
num_jackets = labels_segments.shape[0]
num_labels = np.unique(np.ravel(labels_segments)).size

""" CHANGE THIS IF YOU CHANGE NUMBER OF FEATURES """
if feature_set == 'basic':
    num_features = 5
elif feature_set == 'basic_and_middle':
    num_features = 7
elif feature_set == 'full':
    num_features = 12
else:
    raise ValueError(
        "Feature set option {!r} not available. Choose one of: 'basic', 'basic_and_middle', 'full'".format(feature_set)
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
                       s.x0, s.y0, s.x1, s.y1, np.sqrt((s.x0norm - s.x1norm) ** 2 + (s.y0norm - s.y1norm) ** 2)
        X[i, j, 0:num_features] = features

print('X, Y done')

""" you can add some noise to the features """
if add_gaussian_noise_to_features:
    print('Noise sigma {}'.format(sigma_noise))
    X = X + np.random.normal(0.0, sigma_noise, size=X.size).reshape(np.shape(X))

"""
DEFINE HERE YOUR GRAPHICAL MODEL AND CHOOSE ONE LEARNING METHOD
(OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM)
"""
model = ChainCRF()
try:
    ssvm_class = structured_learning_options[structured_learning_algorithm]
except KeyError:
    raise Exception('{!r} structured learning model not supported. Use one of: {!r}'.format(
        structured_learning_algorithm,
        structured_learning_options.keys(),
    ))
ssvm = ssvm_class(model=model, C=C)

""" LINEAR SVM """
svm = LinearSVC(C=C)

""" 
Compare SVM with S-SVM doing k-fold cross validation, k=5, see scikit-learn.org 
"""
n_folds = 5
""" with 5 in each fold we have 4 jackets for testing, 19 for training, 
with 23 we have leave one out : 22 for training, 1 for testing"""
scores_crf = np.zeros(n_folds)
scores_svm = np.zeros(n_folds)
training_time_crf = np.zeros(n_folds)
training_time_svm = np.zeros(n_folds)
wrong_segments_crf = []
wrong_segments_svm = []

kf = KFold(num_jackets, n_folds=n_folds)
for fold, (train_index, test_index) in enumerate(kf):
    print(' ')
    print('train index {}'.format(train_index))
    print('test index {}'.format(test_index))
    print('{} jackets for training, {} for testing'.format(len(train_index), len(test_index)))
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    """ YOUR S-SVM TRAINING CODE HERE """
    start = time.time()
    ssvm.fit(X_train, Y_train)
    end = time.time()
    print('CRF training time: {:.2f} s'.format(end - start))
    training_time_crf[fold] = end - start

    """ LABEL THE TESTING SET AND PRINT RESULTS """
    crf_score = ssvm.score(X_test, Y_test)
    print("Test score with chain CRF: {}".format(crf_score))
    scores_crf[fold] = crf_score

    Y_pred = np.array(ssvm.predict(X_test))
    wrong_segments_array = Y_pred != Y_test
    wrong_segments_crf.append(np.count_nonzero(wrong_segments_array))

    # figure showing the result of classification of segments for
    # each jacket in the testing part of present fold """
    if plot_labeling:
        for ti, pred in zip(test_index, Y_pred):
            print(ti)
            print(pred)
            s = segments[ti]
            plot_segments(s, caption='SSVM predictions for jacket ' + str(ti + 1),
                          labels_segments=pred)

    """ YOUR LINEAR SVM TRAINING CODE HERE """
    X_train_svm, X_test_svm, Y_train_svm, Y_test_svm = np.vstack(X_train), np.vstack(X_test), \
                                                       np.hstack(Y_train), np.hstack(Y_test)
    start = time.time()
    svm.fit(X_train_svm, Y_train_svm)
    end = time.time()
    print('LinearSVM training time: {:.2f} s'.format(end - start))
    training_time_svm[fold] = end - start

    """ LABEL THE TESTING SET AND PRINT RESULTS """
    svm_score = svm.score(X_test_svm, Y_test_svm)
    print("Test score with linear SVM: {}".format(svm_score))
    scores_svm[fold] = svm_score
    Y_pred_svm = svm.predict(X_test_svm)
    wrong_segments_array = Y_pred_svm != Y_test_svm
    wrong_segments_svm.append(np.count_nonzero(wrong_segments_array))

    if plot_labeling:
        for ti, pred in zip(test_index, Y_pred_svm):
            print(ti)
            print(pred)
            s = segments[ti]
            plot_segments(s, caption='LinearSVM predictions for jacket ' + str(ti + 1),
                          labels_segments=pred)

"""
Global results
"""
total_segments = num_jackets * num_segments_per_jacket
wrong_segments_crf = np.array(wrong_segments_crf)
wrong_segments_svm = np.array(wrong_segments_svm)
print('Results per fold ')
print('Scores CRF : {}'.format(scores_crf))
print('Scores SVM : {}'.format(scores_svm))
print('Wrongs CRF : {}'.format(wrong_segments_crf))
print('Wrongs SVM : {}'.format(wrong_segments_svm))
print('Training time CRF (s): {}'.format(training_time_crf))
print('Training time SVM (s): {}'.format(training_time_svm))
print(' ')
print('Final score CRF: {}, {} wrong labels in total out of {}'.format(
    1.0 - wrong_segments_crf.sum() / float(total_segments),
    wrong_segments_crf.sum(),
    total_segments
))
print('Final score SVM: {}, {} wrong labels in total out of {}'.format(
    1.0 - wrong_segments_svm.sum() / float(total_segments),
    wrong_segments_svm.sum(),
    total_segments)
)
print('Average training time CRF: {:.3f} s'.format(np.mean(training_time_crf)))
print('Average training time SVM: {:.3f} s'.format(np.mean(training_time_svm)))

if plot_coefficients:
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

    """ SHOW IMAGE OF THE LEARNED UNARY COEFFICIENTS, size (num_labels, num_features)"""
    """ use matshow() and colorbar()"""
    unary_weights = ssvm.w[:num_segments_classes * num_features].reshape(num_segments_classes, num_features)
    plt.matshow(unary_weights)
    plt.colorbar()
    plt.title("Transition parameters of the chain CRF.")
    plt.xticks(np.arange(num_features), features_names[:num_features])
    plt.yticks(np.arange(num_segments_classes), name_of_labels)
    plt.show()

    """ SHOW IMAGE OF PAIRWISE COEFFICIENTS size (num_labels, num_labels)"""
    pairwise_weights = ssvm.w[num_segments_classes * num_features:].reshape(num_segments_classes, num_segments_classes)
    plt.matshow(pairwise_weights)
    plt.colorbar()
    plt.title("Transition parameters of the chain CRF.")
    plt.xticks(np.arange(num_segments_classes), name_of_labels)
    plt.yticks(np.arange(num_segments_classes), name_of_labels)
    plt.show()
