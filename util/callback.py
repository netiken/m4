from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        pl_module.log("step", float(trainer.current_epoch))


class MyCallback(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def training_epoch_end(self, outputs):
        loss_mean = outputs.mean()
        self.log("training_loss_epoch", loss_mean)
