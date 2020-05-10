import argparse
import configparser
from sia_auto_generation.sia_auto_generation import SiaAutoGeneration


def parse_args():
    parser = argparse.ArgumentParser(description='bio-sia-auto-generation')
    parser.add_argument('--num_sia_generation', type=int, default=20)
    parser.add_argument('--cuda_device', type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cuda_device = False if args.cuda_device == -1 else True

    config = configparser.ConfigParser()
    config.read('./sia_auto_generation/conf/sia_configuration.conf')
    sia_auto_generation = SiaAutoGeneration(config, cuda_device)
    sia_auto_generation.create_dataset(args.num_sia_generation)
