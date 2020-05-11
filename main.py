import argparse
import configparser
from sia_auto_generation.sia_auto_generation import SiaAutoGeneration


def parse_args():
    parser = argparse.ArgumentParser(description='bio-sia-auto-generation')
    parser.add_argument('--num_sia_generation', type=int, default=20)
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--manual_input', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cuda_device = False if args.cuda_device == -1 else True

    config = configparser.ConfigParser()
    config.read('./sia_auto_generation/conf/sia_configuration.conf')
    sia_auto_generation = SiaAutoGeneration(config, cuda_device)

    if args.manual_input:
        sentence_4 = input('Enter Sentence 4 : ')
        sia_auto_generation.create_data(sentence_4)
    else:
        sia_auto_generation.create_dataset(args.num_sia_generation)
