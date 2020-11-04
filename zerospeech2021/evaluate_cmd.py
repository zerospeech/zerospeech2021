from pathlib import Path
import click


def eval_lexical(dataset_location: Path, submission_location: Path, output: Path, _sets):
    from zerospeech2021.lexical import evaluate

    if 'dev' in _sets:
        gold_file = dataset_location / 'lexical' / 'dev' / 'gold.csv'
        submission_file = submission_location / 'lexical_dev.txt'

        by_pair, by_frequency, by_length = evaluate(
            gold_file, submission_file
        )

        by_pair.to_csv(
            output / 'dev_score_lexical_by_pair.csv',
            index=False, float_format='%.4f'
        )
        by_frequency.to_csv(
            output / 'dev_score_lexical_by_frequency.csv',
            index=False, float_format='%.4f'
        )
        by_length.to_csv(
            output / 'dev_score_lexical_by_length.csv',
            index=False, float_format='%.4f'
        )

    if 'test' in _sets:
        gold_file = dataset_location / 'lexical' / 'test' / 'gold.csv'
        submission_file = submission_location / 'lexical_test.txt'

        by_pair, by_frequency, by_length = evaluate(
            gold_file, submission_file
        )

        by_pair.to_csv(
            output / 'test_score_lexical_by_pair.csv',
            index=False, float_format='%.4f'
        )
        by_frequency.to_csv(
            output / 'test_score_lexical_by_frequency.csv',
            index=False, float_format='%.4f'
        )
        by_length.to_csv(
            output / 'test_score_lexical_by_length.csv',
            index=False, float_format='%.4f'
        )


def eval_semantic(dataset_location: Path, submission_location: Path, output: Path, _sets):
    # from zerospeech2021.semantic import evaluate
    raise NotImplementedError()


def eval_syntactic(dataset_location: Path, submission_location: Path, output: Path, _sets):
    from zerospeech2021.syntactic import evaluate

    if 'dev' in _sets:
        gold_file = dataset_location / 'syntactic' / 'dev' / 'gold.csv'
        submission_file = submission_location / 'syntactic_dev.txt'

        by_pair, by_type = evaluate(
            gold_file, submission_file
        )
        by_pair.to_csv(
            output / 'dev_score_syntactic_by_pair.csv',
            index=False, float_format='%.4f'
        )
        by_type.to_csv(
            output / 'dev_score_syntactic_by_type.csv',
            index=False, float_format='%.4f'
        )

    if 'test' in _sets:
        gold_file = dataset_location / 'syntactic' / 'test' / 'gold.csv'
        submission_file = submission_location / 'syntactic_test.txt'

        by_pair, by_type = evaluate(
            gold_file, submission_file
        )

        by_pair.to_csv(
            output / 'test_score_syntactic_by_pair.csv',
            index=False, float_format='%.4f'
        )
        by_type.to_csv(
            output / 'test_score_syntactic_by_type.csv',
            index=False, float_format='%.4f'
        )


def eval_phonetic(dataset_location: Path, submission_location: Path, output: Path, _sets):
    from zerospeech2021.phonetic import evaluate

    abx_data = dataset_location / 'acoustic' / 'abx_features'

    if 'dev' in _sets:
        features_location = submission_location / 'phonetic' / 'dev'
        evaluate(features_location, abx_data, output, 'dev')

    if 'test' in _sets:
        features_location = submission_location / 'phonetic' / 'test'
        evaluate(features_location, abx_data, output, 'test')


@click.command()
@click.argument('dataset', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('submission', type=click.Path(file_okay=True, dir_okay=True, exists=True))
@click.option('-o', '--output-directory', 'output', type=click.Path(file_okay=False, dir_okay=True, exists=True),
              help="Location to output results")
@click.option('--dev/--no-dev', default=True, help="Evaluate dev set (dev-clean, dev-other)",
              show_default=True)
@click.option('--test/--no-test', default=False, help="Evaluate test set (test-clean, test-other)",
              show_default=True)
@click.option('--lexical/--no-lexical', default=True, help="Evaluate lexical task",
              show_default=True)
@click.option('--semantic/--no-semantic', default=True, help="Evaluate semantic task",
              show_default=True)
@click.option('--syntactic/--no-syntactic', default=True, help="Evaluate syntactic task",
              show_default=True)
@click.option('--phonetic/--no-phonetic', default=True, help="Evaluate phonetic task",
              show_default=True)
def evaluate(**kwargs):
    """ Evaluate submission """
    dataset_location = Path(kwargs.get('dataset'))
    submission_location = Path(kwargs.get('submission'))

    _sets = []
    if kwargs.get("dev"): _sets.append('dev')
    if kwargs.get("test"): _sets.append('dev')

    output = Path(kwargs.get("output", 'evaluation_output'))
    if not output.is_dir():
        output.mkdir(exist_ok=True, parents=True)

    if kwargs.get("lexical"):
        eval_lexical(dataset_location, submission_location, output, _sets)

    if kwargs.get("semantic"):
        eval_semantic(dataset_location, submission_location, output, _sets)

    if kwargs.get("syntactic"):
        eval_syntactic(dataset_location, submission_location, output, _sets)

    if kwargs.get("phonetic"):
        eval_phonetic(dataset_location, submission_location, output, _sets)


if __name__ == '__main__':
    evaluate()
