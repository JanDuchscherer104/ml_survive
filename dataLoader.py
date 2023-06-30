import os
from random import choices, sample
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split


class PlantDiseaseDataLoader:
    def __init__(
        self,
        width=128,
        height=None,
        modes=["leaf_type"],
        src="PlantVillage-Dataset/raw/segmented",
        verbose=False,
    ):
        self._src_dir = os.path.join(os.getcwd(), src)
        self._width = width
        self._height = height if height else width
        self._modes: List[str] = modes

        # data that will be loaded by loadDataset()
        self._data: List[np.array] = []
        self._labels: List[List[int]] = []
        self._mean = None
        self._scale = None

        # helper attributes to parse different classes from the directory names
        self._dir_list = sorted(os.listdir(self._src_dir))

        self._dir_to_label_name: List[Dict[str, str]] = [
            self._get_labels_from_dirs(mode) for mode in self._modes
        ]
        self._label_name_to_id: List[Dict[str, int]] = [
            {name: i for i, name in enumerate(set(dir_to_label_name.values()))}
            for dir_to_label_name in self._dir_to_label_name
        ]

        self.verbose = verbose
        if self.verbose:
            print("DataLoader initialized with:")
            print("src_dir:", self._src_dir)
            print("width:", self._width)
            print("height:", self._height)
            print("mode:", self._modes)
            print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    def loadDataset(self, drop_out=0.2):
        labels = []
        # read all images in PATH, resize and write to DESTINATION_PATH
        for subdir in self._dir_list:
            current_path = os.path.join(self._src_dir, subdir)
            for file in os.listdir(current_path):
                if choices([True, False], [1 - drop_out, drop_out])[0]:
                    if file[-3:] in {"jpg", "png"}:
                        im = cv2.imread(os.path.join(current_path, file))
                        im = np.array(
                            cv2.resize(
                                im,
                                (self._width, self._height),
                                interpolation=cv2.INTER_AREA,
                            )
                        )
                        for mode in self._modes:
                            labels.append(self._get_id_from_dir(subdir))
                            self._data.append(im)

        self._labels = labels
        if self.verbose:
            print(f"Loaded {len(self._data)} images")

    def getSplitDataset(self, valid_size=0.2, test_size=0.2, random_state=42):
        """

        Args:
            valid_size (float, optional): _description_. Defaults to 0.2.
            test_size (float, optional): _description_. Defaults to 0.2.
            random_state (int, optional): _description_. Defaults to 42.

        Returns:
            Tuple[np.array()]: The labels (y_* = List[#modes x #samples]])
        """
        X_train = []
        X_valid = []
        X_test = []
        y_train = []
        y_valid = []
        y_test = []

        X_train, X_test, y_train, y_test = train_test_split(
            self._data,
            self._labels,
            test_size=test_size + valid_size,
            random_state=random_state,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test,
            y_test,
            test_size=test_size / (test_size + valid_size),
            random_state=random_state,
        )

        # normalize data
        self._mean = np.mean(X_train)
        self._scale = np.std(X_train)
        X_train = (X_train - self._mean) / self._scale
        X_valid = (X_valid - self._mean) / self._scale
        X_test = (X_test - self._mean) / self._scale

        return (
            np.array(X_train),
            np.array(X_valid),
            np.array(X_test),
            np.array(y_train),
            np.array(y_valid),
            np.array(y_test),
        )

    def augment_data(self, augmentation_factor=0.5):
        prev_len = len(self._labels)

        # Define the augmentation sequence
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontal flips
                iaa.Crop(percent=(0, 0.1)),  # random crops
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8),
                ),
            ],
            random_order=True,
        )

        for idx, image in enumerate(self._data):
            # Perform the augmentation for the current image
            if choices([True, False], [augmentation_factor, 1 - augmentation_factor])[
                0
            ]:
                self._data.append(seq.augment_image(image))
                self._labels.append(self._labels[idx])

        if self.verbose:
            print(
                f"{len(self._data) - prev_len} augmented images have been added to the data set!"
            )

            nrow = 5
            ncol = 5
            _, axs = plt.subplots(nrow, ncol, figsize=(12, 12))
            axs = axs.flatten()
            for img_label, ax in zip(
                list(zip(self._data, self._labels))[prev_len : prev_len + nrow * ncol],
                axs,
            ):
                ax.set_axis_off()
                ax.set_title(img_label[1], fontsize=8)
                ax.imshow(cv2.cvtColor(img_label[0], cv2.COLOR_BGR2RGB))
            plt.show()

    def get_data(self):
        return self._data, self._labels

    def get_mean_scale(self):
        return self._mean, self._scale

    def get_uniqe_label_names(self):
        return list(self._dir_to_label_name.values())

    def plot_data_statistics(self):
        if self._data is None:
            self.loadDataset()
        if self.verbose:
            for dir_to_label_name in self._dir_to_label_name:
                print("Dict{dir_name: label_name}:\n", dir_to_label_name, "\n")
            for label_name_to_id in self._label_name_to_id:
                print("Dict{label_name: label_id}:\n", label_name_to_id)

        for mode, named_labels in zip(self._modes, self.get_label_names()):
            # plot number of images per class
            plt.set_loglevel("info")
            plt.suptitle(f"number of images for classification mode {mode}")
            plot_bar(list(named_labels), loc=-0.0)  # uniques
            plt.legend(["{0} images (complete)".format(len(list(named_labels)))])
            plt.show()

            # inspect the dataset by displaying some random samples.
            nrow = 5
            ncol = 5
            _, axs = plt.subplots(nrow, ncol, figsize=(12, 12))
            axs = axs.flatten()
            for img_label, ax in zip(
                sample(list(zip(self._data, named_labels)), nrow * ncol), axs
            ):
                ax.set_axis_off()
                ax.set_title(img_label[1], fontsize=8)
                ax.imshow(cv2.cvtColor(img_label[0], cv2.COLOR_BGR2RGB))
            plt.show()

    def _get_labels_from_dirs(self, mode) -> Dict[str, str]:
        if mode == "leaf_type":
            return {name: name.partition("___")[0] for name in self._dir_list}
        if mode == "disease":
            return {name: name.partition("___")[2] for name in self._dir_list}
        if mode == "healthy":
            return {
                name: "healthy" if name.partition("___")[2] == "healthy" else "infected"
                for name in self._dir_list
            }
        if mode == "full":
            return {name: name for name in self._dir_list}

    def _get_id_from_dir(self, dir):
        ret = []
        for dir_to_label_name, label_name_to_id in zip(
            self._dir_to_label_name, self._label_name_to_id
        ):
            ret.append(label_name_to_id[dir_to_label_name[dir]])
        return ret

    def get_label_names(self) -> List[List[str]]:
        ret = []
        for label_name_to_id, labels in zip(self._label_name_to_id, zip(*self._labels)):
            ret.append([list(label_name_to_id.keys())[id] for id in labels])
        return ret

    def get_class_names(self):
        return list(
            set(dir_to_label_name.values())
            for dir_to_label_name in self._dir_to_label_name
        )


def plot_bar(y, loc=0.5, relative=False):
    """
    Bar plot
    Input:
        loc -- relative horizontal location of bar
        relative -- bars normalized if True
    """
    width = 0.35

    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]

    if relative:
        # plot as a percentage
        counts = 100 * counts[sorted_index] / len(y)
        ylabel_text = "% count"
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = "count"

    xtemp = np.arange(len(unique))

    plt.bar(xtemp + loc * width, counts, align="center", alpha=0.7, width=width)
    plt.xticks(xtemp, unique, rotation=90)
    plt.xlabel("Classes")
    plt.ylabel(ylabel_text)
