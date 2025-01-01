def load_model(self, checkpoint_path):
        model_config = self.model_config
        training_config = self.training_config
        dataset_config = self.dataset_config
        if self.model_name == "lstm":
            model = FlowSimLstm.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device,
                n_layer=model_config["n_layer"],
                gcn_n_layer=model_config["gcn_n_layer"],
                loss_fn_type=model_config["loss_fn_type"],
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                hidden_size=model_config["hidden_size"],
                gcn_hidden_size=model_config["gcn_hidden_size"],
                dropout=model_config["dropout"],
                enable_val=training_config["enable_val"],
                enable_dist=training_config["enable_dist"],
                input_size=model_config["input_size"],
                output_size=1,
                enable_bidirectional=model_config.get("enable_bidirectional", False),
                enable_positional_encoding=model_config.get(
                    "enable_positional_encoding", False
                ),
                enable_gnn=model_config.get("enable_gnn", False),
                enable_lstm=model_config.get("enable_lstm", False),
                current_period_len_idx=dataset_config.get(
                    "current_period_len_idx", None
                ),
                enable_lstm_in_gnn=model_config.get("enable_lstm_in_gnn", False),
                enable_link_state=model_config.get("enable_link_state", False),
                enable_flowsim_diff=dataset_config.get("enable_flowsim_diff", False),
                enable_remainsize=dataset_config.get("enable_remainsize", False),
                enable_queuelen=dataset_config.get("enable_queuelen", False),
                enable_path=dataset_config.get("enable_path", False),
                enable_topo=dataset_config.get("enable_topo", False),
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        if self.enable_link_state:
            return (
                model.gcn_layers,
                model.lstmcell_rate,
                model.lstmcell_time,
                model.output_layer,
                model.lstmcell_rate_link,
                model.lstmcell_time_link,
            )