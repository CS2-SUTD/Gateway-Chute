from chute import Chute
import argparse


def get_arguments():
    config_path = "config.ini"
    source = "data/sample.mp4"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=config_path)
    parser.add_argument("-s", "--source", type=str, default=source)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    chute = Chute(config_path=args.config)
    chute.start(args.source)
