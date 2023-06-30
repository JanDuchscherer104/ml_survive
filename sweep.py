import tensorflow as tf
import yaml

import wandb

# # Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "epochs": {"values": [100]},
        "batch_size": {"values": [8, 16, 32, 64]},
        "lr": {"max": 0.1, "min": 0.001},
        "trainable_layers": {"values": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "epsilon": {"max": 1.0, "min": 0.001},
        "dense": {"values": [1024, 2048, 4096, 8192, 16384]},
        "dropout": {"max": 0.5, "min": 0.1},
        "data_augmentation": {"max": 10.0, "min": 0.1},
        "addtional_layers": {"values": [0, 1, 2, 4, 8]},
    },
}
# #
# # with open("sweep.yaml") as f:
# #     sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
#
# # Initialize sweep by passing in config. (Optional) Provide a name of the project.

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="ml_survive", entity="hm-muc")


def main():
    from dataLoader import PlantDiseaseDataLoader
    from model import Model

    # note that we define values from `wandb.config` instead
    # of defining hard values
    run = wandb.init()

    wandb.config.width = 75
    wandb.config.data_dropout = 0
    wandb.config.mode = "full"

    lr = wandb.config.lr
    bs = wandb.config.batch_size
    dense = wandb.config.dense
    dropout = wandb.config.dropout
    epochs = wandb.config.epochs
    epsilon = wandb.config.epsilon
    trainable_layers = wandb.config.trainable_layers
    data_augmentation = wandb.config.data_augmentation

    width = wandb.config.width
    data = PlantDiseaseDataLoader(width=width, modes=[wandb.config.mode])
    data.loadDataset(drop_out=wandb.config.data_dropout)
    data.augment_data(augmentation_factor=data_augmentation)
    X_Train, X_Valid, X_Test, y_Train, y_Valid, y_Test = data.getSplitDataset()
    y_Train = y_Train[:, 0]
    y_Valid = y_Valid[:, 0]
    y_Test = y_Test[:, 0]

    M_leaf_type = Model(
        width, width, "InceptionV3", n_classes=38, class_names=data.get_class_names()[0]
    )
    M_leaf_type.get_model(
        lr=lr,
        trainable_layers=trainable_layers,
        epsilon=epsilon,
        dense=dense,
        dropout=dropout,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    leaf_type_history = M_leaf_type.train_model(
        X_Train,
        y_Train,
        X_Valid,
        y_Valid,
        epochs=epochs,
        batch_size=bs,
        callbacks=[reduce_lr, early_stopping],
    )
    wandb.log(
        {
            "epoch": len(leaf_type_history.history["loss"]),
            "val_accuracy": leaf_type_history.history["val_accuracy"][-1],
            "test_accuracy": M_leaf_type.test_accuracy(X_Test, y_Test),
        }
    )
    wandb.finish()


# Start sweep job.
wandb.agent("5adutz1u", function=main, count=1, project="ml_survive", entity="hm-muc")
