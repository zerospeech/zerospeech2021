import json
import sys
from pathlib import Path

import click

from zerospeech2021.leaderboard import get_semantic_size, ZeroSpeechSubmission


def create(submission_location: Path, dataset_location, score_location: Path,
           user_meta, leaderboard_file: Path):
    """ Function that builds a leaderboard entry from the computed scores of evaluation

    ARGS:
        submission_location<PathDir>: location to the submission entry files (as described in ...)
        dataset_location<PathDir>: location of the test set
        score_location<PathDir>: location of the scores computed by evaluation
        user_meta<PathFile>: file containing platform metadata (user, submission date etc.)
        leaderboard_file<PathFile>: location & name to write result file
    """
    print("Building leaderboard entry from scores...")
    semantic_size = get_semantic_size(dataset_location)

    if not submission_location.is_dir():
        print("SUBMISSION folder not found", file=sys.stderr)
        sys.exit(-1)

    if not dataset_location.is_dir():
        print("DATASET folder not found", file=sys.stderr)
        sys.exit(-1)

    if not score_location.is_dir():
        print("SCORE folder not found", file=sys.stderr)
        sys.exit(-1)

    if leaderboard_file.is_file():
        print(f"WARNING: leaderboard specified already exists: [OVERWRITING] {leaderboard_file}", file=sys.stderr)

    subs = ZeroSpeechSubmission(
        submission_location=submission_location, external_meta_file=user_meta,
        _semantic_size=semantic_size, score_location=score_location,
    )

    leaderboard_file = leaderboard_file.with_suffix(".json")
    with leaderboard_file.open('w') as fp:
        json.dump(subs.leaderboard(), fp, indent=4)
    print(f"\t> Wrote {leaderboard_file}")


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('submission', type=Path)
@click.argument('dataset', type=Path)
@click.argument('scores', type=Path)
@click.option('-u', '--user-meta', type=Path, help="Location of platform metadata")
@click.option('-o', '--output-file', type=Path, help="Location & name of the leaderboard file")
def leaderboard(submission: Path, dataset: Path, scores: Path, user_meta, output_file):
    """ CLI wrapper to build leaderboard entry """
    try:
        create(submission, dataset, scores, user_meta, output_file)
    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)
