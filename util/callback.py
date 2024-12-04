from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from .consts import balance_len_bins_list


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)
        if trainer.current_epoch == 0:
            trainer.datamodule.switch_to_other_epochs_logic()

        if pl_module.current_period_len_idx is not None:
            avg_loss = trainer.callback_metrics.get("train_loss_sync")

            if avg_loss is not None:
                # Logic to switch flow periods based on the average loss
                if pl_module.current_period_len_idx < len(balance_len_bins_list) - 1:
                    if avg_loss.item() < 0.02:  # Loss threshold for switching periods
                        pl_module.current_period_len_idx += 1
                        trainer.datamodule.switch_to_next_flow_period(
                            pl_module.current_period_len_idx
                        )
                else:
                    pl_module.current_period_len_idx = None
                    trainer.datamodule.switch_to_next_flow_period(
                        pl_module.current_period_len_idx
                    )

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

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # do something with all training_step outputs, for example:
    # epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
    # pl_module.log("training_epoch_mean", epoch_mean)
    # free up the memory
    # print("clear on_train_batch_end")
    # pl_module.training_step_outputs.clear()
    # gc.collect()
    # torch.cuda.empty_cache()
