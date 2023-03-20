# pip install black ; black .
from model import NN
from dataset import MnistDataModule
import config
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    sweep_id = wandb.sweep(config.sweep_config, project='MNIST')

    def sweep_iteration():
        # set up W&B logger
        wandb.init()    # required to have access to `wandb.config`
        wandb_logger = WandbLogger(project='MNIST')

        # setup data
        dm = MnistDataModule(
            data_dir=config.DATA_DIR,
            batch_size=wandb.config.batch_size,
            num_workers=config.NUM_WORKERS,
        )

        # setup model - refer to sweep parameters with wandb.config
        model = NN(
            input_size=config.INPUT_SIZE,
            n_layer_1=wandb.config.n_layer_1,
            n_layer_2=wandb.config.n_layer_2,
            learning_rate=wandb.config.lr,
            num_classes=config.NUM_CLASSES,
        )

        # setup Trainer
        trainer = pl.Trainer(
            accelerator=config.ACCELERATOR,
            logger=wandb_logger,            # W&B integration
            devices=config.DEVICES,
            max_epochs=config.MAX_EPOCHS,    # number of epochs
            precision=config.PRECISION,
            callbacks=[
                ModelCheckpoint(monitor='val_loss', save_weights_only=True, mode='min'),
                EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min'),
                LearningRateMonitor("epoch"),
            ]
        )

        # train
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)

    wandb.agent(sweep_id, function=sweep_iteration)
    wandb.finish()
