from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from util.dataset import PathDataModulePerFlow
from util.arg_parser import create_config
from util.func import (
    fix_seed,
    create_logger,
)
from util.model import (
    FlowSimLstm,
    FlowSimTransformer
)
from util.callback import OverrideEpochStepCallback
import logging, os
import torch
import yaml
import numpy as np

torch.set_float32_matmul_precision(precision="high")
args = create_config()
config_file = args.train_config if args.mode == "train" else args.test_config
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]

shard = dataset_config["shard"]
fix_seed(shard)

# get dataset configurations
shard_list_config = dataset_config["shard_list"]
shard_list = sorted(
    np.random.choice(
        np.arange(shard_list_config[0], shard_list_config[1]),
        size=shard_list_config[2],
        replace=False,
    )
)
n_flows_list = dataset_config["n_flows_list"]
n_hosts_list = dataset_config["n_hosts_list"]
sample_list_config = dataset_config["sample_list"]
sample_list = sorted(
    np.random.choice(
        np.arange(sample_list_config[0], sample_list_config[1]),
        size=sample_list_config[2],
        replace=False,
    )
)

lr = dataset_config["lr"]
note_str = f"{args.note}_" if args.note else ""
program_name = f"{note_str}shard{len(shard_list)}_nflows{len(n_flows_list)}_nhosts{len(n_hosts_list)}_nsamples{len(sample_list)}_lr{lr}Gbps"
override_epoch_step_callback = OverrideEpochStepCallback()
dir_output=args.dir_output
dir_input=args.dir_input

if args.mode == "train":
    tb_logger = TensorBoardLogger(dir_output, name=program_name)

    # configure logging at the root level of Lightning
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    model_name = "bs{}_lr{}_{}_nlayer{}".format(
        training_config["batch_size"],
        training_config["learning_rate"],
        model_config["model_name"],
        model_config["n_layer"],
    )
    create_logger(os.path.join(tb_logger.log_dir, f"{model_name}.log"))
    logging.info(f"Save to: {tb_logger.log_dir}")
    logging.info(args)

    enable_dist = training_config["enable_dist"]
    enable_val = training_config["enable_val"]
    if enable_dist:
        from pytorch_lightning.strategies import DDPStrategy

        logging.info(
            f"gloo: {torch.distributed.is_gloo_available()}, nccl: {torch.distributed.is_nccl_available()}"
        )
        
        ddp_strategy = DDPStrategy(
            process_group_backend="gloo", 
            # find_unused_parameters=True
        )
        
    with open(f"{tb_logger.log_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    datamodule = PathDataModulePerFlow(
        dir_input=dir_input,
        shard_list=shard_list,
        n_flows_list=n_flows_list,
        n_hosts_list=n_hosts_list,
        sample_list=sample_list,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        train_frac=dataset_config["train_frac"],
        dir_output=tb_logger.log_dir,
        lr=lr,
        topo_type=dataset_config.get("topo_type", ""),
        output_type=dataset_config.get("output_type", "fctSldn"),
    )

    # Init checkpointer
    if enable_val:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_sync" if enable_dist else "val_loss",
            dirpath=f"{tb_logger.log_dir}/checkpoints",
            filename="model-{epoch:03d}-{step:05d}-{val_loss_sync:.2f}"
            if enable_dist
            else "model-{epoch:03d}-{step:05d}-{val_loss:.2f}",
            save_top_k=5,
            save_last=True,
            # every_n_train_steps=5,
            mode="min",
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss_sync" if enable_dist else "train_loss",
            dirpath=f"{tb_logger.log_dir}/checkpoints",
            filename="model-{epoch:03d}-{step:05d}-{train_loss_sync:.2f}"
            if enable_dist
            else "model-{epoch:03d}-{step:05d}-{train_loss:.2f}",
            save_top_k=5,
            save_last=True,
            # every_n_train_steps=5,
            mode="min",
        )

    callbacks = [checkpoint_callback, override_epoch_step_callback]

    trainer = Trainer(
        logger=[tb_logger],
        callbacks=callbacks,
        max_epochs=training_config["n_epochs"],
        accelerator="gpu",
        devices=training_config["gpu"],
        strategy=ddp_strategy if enable_dist else "auto",
        default_root_dir=dir_output,
        # log_every_n_steps=args.n_epochs_every_log,
        log_every_n_steps=1,
        val_check_interval=1.0,
        # fast_dev_run=args.debug,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # enable_progress_bar=True,
    )
    model_name = model_config["model_name"]
    if model_name == "lstm":
        model = FlowSimLstm(
            n_layer=model_config["n_layer"],
            loss_fn_type=model_config["loss_fn_type"],
            learning_rate=training_config["learning_rate"],
            batch_size=training_config["batch_size"],
            hidden_size=model_config["hidden_size"],
            enable_val=enable_val,
            enable_dist=enable_dist,
            input_size=2,
            output_size=1,
            enable_bidirectional=model_config.get("enable_bidirectional", False),
        )
    elif model_name == "transformer":
        model = FlowSimTransformer(
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            dropout=model_config["dropout"],
            compile=model_config["compile"],
            loss_fn_type=model_config["loss_fn_type"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            betas=training_config["betas"],
            batch_size=training_config["batch_size"],
            enable_position=model_config["enable_position"],
            enable_val=enable_val,
            enable_dist=enable_dist,
        )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
else:
    DEVICE = torch.device(training_config["gpu"][0])
    dir_train = (
        f"{dir_output}/{program_name}/version_{args.version_id}"
    )
    print(f"load model: {dir_train}")

    tb_logger = TensorBoardLogger(dir_train, name="test")

    # configure logging at the root level of Lightning
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    create_logger(os.path.join(tb_logger.log_dir, f"console.log"))
    logging.info(f"Save to: {tb_logger.log_dir}")
    logging.info(args)

    datamodule = PathDataModulePerFlow(
        dir_input=dir_input,
        shard_list=shard_list,
        n_flows_list=n_flows_list,
        n_hosts_list=n_hosts_list,
        sample_list=sample_list,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        train_frac=dataset_config["train_frac"],
        dir_output=dir_train,
        # customized config
        lr=dataset_config["lr"],
        topo_type=dataset_config.get("topo_type", ""),
        output_type=dataset_config.get("output_type", "fctSldn"),
        mode=args.mode,
        test_on_train=args.test_on_train,
        test_on_empirical=args.test_on_empirical,
        test_on_manual=args.test_on_manual,
    )

    callbacks = [override_epoch_step_callback]

    trainer = Trainer(
        logger=[tb_logger],
        callbacks=callbacks,
        # max_epochs=training_config["n_epochs"],
        accelerator="gpu",
        devices=training_config["gpu"],
        # strategy=ddp_strategy if enable_dist else "auto",
        default_root_dir=dir_train,
        log_every_n_steps=1,
        # val_check_interval=0.1
        # fast_dev_run=args.debug,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # enable_progress_bar=False,
    )
    model_name = model_config["model_name"]
    if model_name == "lstm":
        model = FlowSimLstm.load_from_checkpoint(
            f"{dir_train}/checkpoints/last.ckpt",
            map_location=DEVICE,
            n_layer=model_config["n_layer"],
            loss_fn_type=model_config["loss_fn_type"],
            learning_rate=training_config["learning_rate"],
            batch_size=training_config["batch_size"],
            hidden_size=model_config["hidden_size"],
            enable_val=training_config["enable_val"],
            enable_dist=training_config["enable_dist"],
            input_size=2,
            output_size=1,
            enable_bidirectional=model_config.get("enable_bidirectional", False),
            save_dir=tb_logger.log_dir,
        )
    elif model_name == "transformer":
        model = FlowSimTransformer.load_from_checkpoint(
            f"{dir_train}/checkpoints/last.ckpt",
            map_location=DEVICE,
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            dropout=model_config["dropout"],
            compile=model_config["compile"],
            loss_fn_type=model_config["loss_fn_type"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            betas=training_config["betas"],
            batch_size=training_config["batch_size"],
            enable_position=model_config["enable_position"],
            enable_val=training_config["enable_val"],
            enable_dist=training_config["enable_dist"],
            save_dir=tb_logger.log_dir,
        )
    trainer.test(model, datamodule=datamodule)
