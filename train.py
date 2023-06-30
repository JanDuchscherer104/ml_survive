import yaml
import wandb

def main():
    from dataLoader import PlantDiseaseDataLoader
    from model import Model
    # note that we define values from `wandb.config` instead 
    # of defining hard values
    with open("sweep.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    
    wandb.config.width = 75
    wandb.config.data_dropout = 0
    wandb.config.mode = "full"
    
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    dense = wandb.config.dense
    dropout = wandb.config.dropout
    epochs = wandb.config.epochs
    epsilon = wandb.config.epsilon
    trainable_layers = wandb.config.trainable_layers
    
    width = wandb.config.width
    data = PlantDiseaseDataLoader(width=width, modes=[wandb.config.mode], verbose=True)
    data.loadDataset(drop_out=wandb.config.data_dropout)
    X_Train, X_Valid, X_Test, y_Train, y_Valid, y_Test = data.getSplitDataset()
    y_Train = y_Train[:,0]
    y_Valid = y_Valid[:,0]
    y_Test = y_Test[:,0]

    M_leaf_type = Model(width, width, "InceptionV3", 
                        n_classes=38, class_names=data.get_class_names()[0])
    M_leaf_type.get_model(lr=lr, trainable_layers=trainable_layers, 
                          epsilon=epsilon, 
                          dense=dense, dropout=dropout)
    leaf_type_history = M_leaf_type.train_model(X_Train, y_Train, X_Valid, y_Valid,
                                                epochs=epochs, batch_size=bs)
    wandb.log({
    'epoch': epochs, 
    'batch_size': bs,
    'lr': lr,
    'epsilon': epsilon,
    'dense': dense,
    'dropout': dropout,
    'trainable_layers': trainable_layers,
    'val_accuracy': leaf_type_history.history['val_accuracy'][-1],
    'test_accuracy': M_leaf_type.test_accuracy(X_Test, y_Test),
    })
    wandb.finish()
    
if __name__ == "__main__":
    main()