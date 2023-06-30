import json
import os
from typing import Literal

import numpy as np
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from helpers import plot_confusion_matrix


class Model:
    def __init__(
        self,
        width,
        height,
        model_type: Literal["VGG16", "InceptionV3"],
        n_classes: int,
        model_path=None,
        download=None,
        class_names=None,
    ):
        self.width = width
        self.height = height
        self.model_name = model_type
        self.n_classes = n_classes
        self.model = None
        self.history = None
        self.download = download
        self._model_path = model_path
        self.class_names = (
            class_names if class_names else list(str(i) for i in range(n_classes))
        )

        wandb.config = {
            "layer_1": 512,
            "activation_1": "relu",
            "dropout": 0.5,
            "layer_2": n_classes,
            "activation_2": "sigmoid",
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metric": "accuracy",
        }

    def get_model(
        self,
        lr=0.001,
        trainable_layers=100,
        epsilon=1e-7,
        dense=512,
        dropout=0.5,
        addtional_layers=1,
    ):
        if self.model:
            print("Model has already been loaded and trained!")
        elif self._model_path and os.path.exists(self._model_path):
            self.model = keras.models.load_model(self._model_path)
            print("Loaded trained model!")
        elif self.download:
            print("Downloading model from wandb...")
            artifact = self.run.use_artifact(self._model_path, type="model")
            artifact_dir = artifact.download()
            self.model = keras.models.load_model(artifact_dir)
        elif self.model_name == "VGG16":
            base_model = VGG16(
                input_shape=(self.width, self.height, 3),
                include_top=False,  # Leave out the last fully connected layer
                weights="imagenet",
            )
            for layer in base_model.layers:
                layer.trainable = False
        elif self.model_name == "InceptionV3":
            base_model = InceptionV3(
                input_shape=(self.width, self.height, 3),
                weights="imagenet",
                include_top=False,
            )
            for layer in base_model.layers[:trainable_layers]:
                layer.trainable = False
            for layer in base_model.layers[trainable_layers:]:
                layer.trainable = True
        else:
            raise ValueError("Model name is not valid")
        base_model_x = layers.Flatten()(base_model.output)
        for i in range(1, addtional_layers + 1):
            base_model_x = layers.Dense(int(dense / i), activation="relu")(base_model_x)
        base_model_x = layers.Dropout(dropout)(base_model_x)
        base_model_x = layers.Dense(
            self.n_classes, activation=keras.activations.softmax
        )(base_model_x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model_x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        self.model = model
        return self.model

    def train_model(
        self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=64, callbacks=[]
    ):
        # y_* : List[List[int]], where each inner list is a label list

        wandb.config["epoch"] = epochs
        wandb.config["batch_size"] = batch_size
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_valid, y_valid),
            steps_per_epoch=len(X_train) // batch_size,
            validation_steps=len(X_valid) // batch_size,
            callbacks=[WandbMetricsLogger(log_freq=5), *callbacks],
        )
        return self.history

    def plot_history(self):
        if not self.history:
            raise ValueError("Model has not been trained yet")
        # lets plot the validation and train accuracy over the epochs
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.show()

    def test_model(self, X_test, y_test):
        pred_test = self.model.predict(X_test)
        print(
            "Classification accuracy on test set: %0.4f"
            % get_classification_accuracy(y_test, np.array(pred_test))
        )
        return pred_test

    def test_accuracy(self, X_test, y_test):
        pred_test = self.model.predict(X_test)
        return get_classification_accuracy(y_test, np.array(pred_test))

    def confusion_matrix(self, X_test, y_test):
        if not self.history:
            raise ValueError("Model has not been trained yet")
        pred_test = self.model.predict(X_test)
        plot_confusion_matrix(
            y_test, np.array(pred_test), classes=self.class_names, normalize=True
        )

    def save_model(self, model_file_name):
        abs_path = os.path.join(os.getcwd(), "saved_models", model_file_name)
        self.model.save(abs_path)

    def load_model(self, model_file_name):
        abs_path = os.path.join(os.getcwd(), "saved_models", model_file_name)
        self.model = keras.models.load_model(abs_path)

    def save_history(self, history_name):
        abs_path = os.path.join(os.getcwd(), "saved_histories", history_name + ".json")
        json.dump(self.history.history, open(abs_path, "x"))


def get_classification_accuracy(labels, predictions):
    """
    Accuracy and time of classifier on complete test dataset
    Output:
    accuracy -- of classifier on test dataset
    """
    assert len(labels) == len(predictions)
    # in case we have (#smaple x #classes) we have a probability prediction,
    # so we need to find the label with max probability
    if predictions.ndim == 2:
        pred_max = np.argmax(predictions, axis=-1)
    else:
        pred_max = predictions
    accuracy = 100 * sum(a == b for a, b in zip(pred_max, labels)) / len(labels)

    return accuracy
