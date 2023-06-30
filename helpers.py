import itertools
import logging
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# helper functions


def set_logging_level():
    """set logging level (hard_coded)"""
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


def plot_confusion_matrix(
    y_test_pred,
    y_test,
    class_names,
    normalize=True,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):

    assert len(y_test) == len(y_test_pred)

    # in case we have (#smaple x #classes) we have a probability prediction,
    # so we need to find the label with max probability
    if y_test_pred.ndim == 2:
        pred_max = np.argmax(y_test_pred, axis=-1)
    else:
        pred_max = y_test_pred

    cm = metrics.confusion_matrix(y_test, pred_max, normalize="true")

    plt.rcParams.update({"font.size": 20, "figure.dpi": 600})
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def show_classified_samples(images, preds, labels):
    max_pred = np.argmax(preds, axis=-1)
    matches = [1 if i == j else 0 for i, j in zip(max_pred, labels)]
    nrow = 8
    ncol = 8
    _, axs = plt.subplots(nrow, ncol, figsize=(12, 12))
    axs = axs.flatten()
    for img, correct, ax in zip(random.sample(list(images), nrow * ncol), matches, axs):
        img_sample_copy = np.array(img).copy()
        if not correct:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(
            img_sample_copy,
            (0, 0),
            (img_sample_copy.shape[0], img_sample_copy.shape[1]),
            color=color,
            thickness=10,
        )
        # you could also decide to visualize the probability with a text object on each image sample
        # cv2.putText(img_sample_copy, "%0.2f" % pred_val, text_origin, font,
        #                fontScale, color, thickness, cv2.LINE_AA)
        ax.set_axis_off()
        ax.imshow(img_sample_copy)
    plt.show()
