"""Submission Validation for the ZeroSpeech2021 challenge"""

import argparse
import pathlib
import sys

from zerospeech2021 import exception, lexical


def main():
    """CLI for the ZR2021 validation program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('submission_file', type=pathlib.Path)
    parser.add_argument('dataset_directory', type=pathlib.Path)
    args = parser.parse_args()

    try:
        lexical.validate(
            args.submission_file, args.dataset_directory, 'dev')
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


if __name__ == "__main__":
    main()
