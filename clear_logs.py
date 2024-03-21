import os
import glob
import argparse


def clear(directory):
    patterns = ['ema_*', 'opt*']

    # Loop through each pattern
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        for filename in glob.glob(full_pattern):
            os.remove(filename)


def main(args):
    log_dir = os.path.join('./logs', args.exp, 'training')
    clear(log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    args = parser.parse_args()

    main(args)
