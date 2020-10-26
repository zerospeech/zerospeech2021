"""Submission evaluation for the ZeroSpeech2021 challenge"""

import argparse
import pathlib

from zerospeech2021 import lexical


def main():
    """CLI for the ZR2021 evaluation program"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gold_file', type=pathlib.Path)
    parser.add_argument(
        'submission_file', type=pathlib.Path)
    parser.add_argument(
        '-o', '--output-directory', default='.', type=pathlib.Path)
    args = parser.parse_args()

    by_pair, by_frequency = lexical.evaluate(
        args.gold_file, args.submission_file)

    by_pair.to_csv(
        args.output_directory / 'score_lexical_by_pair.csv', index=False)
    by_frequency.to_csv(
        args.output_directory / 'score_lexical_by_frequency.csv', index=False)


if __name__ == '__main__':
    main()
