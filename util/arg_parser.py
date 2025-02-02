import argparse
import datetime, time


def create_config():
    parser = argparse.ArgumentParser(description="Run flow based network simulation")
    # basic config
    parser.add_argument(
        "--train_config",
        type=str,
        default="./config/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default="./config/test_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dir_input",
        help="input directory (dataset)",
        default="/data2/lichenni/path_perflow",
    )
    parser.add_argument(
        "--dir_output",
        help="output directory (checkpotins)",
        default="/data2/lichenni/output_perflow",
    )
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--note", type=str, default="m3")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--test_on_train", action="store_true")
    parser.add_argument("--test_on_empirical", action="store_true")
    parser.add_argument(
        "--version_id",
        type=int,
        default=0,
    )
    parser.add_argument("--shard", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, help="Compute device", default="cuda:1")
    args = parser.parse_args()
    args.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%m%d_%H%M%S"
    )
    return args


if __name__ == "__main__":
    args = create_config()
    print(args)
