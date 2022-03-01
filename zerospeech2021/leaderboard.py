import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml


class LexicalScores:
    """ Class that extracts lexical scores resume from a scores directory """
    # score files
    __dev_pairs = 'score_lexical_dev_by_pair.csv'
    __test_pairs = 'score_lexical_test_by_pair.csv'
    __dev_frequency = 'score_lexical_dev_by_frequency.csv'
    __test_frequency = 'score_lexical_test_by_frequency.csv'
    __dev_length = 'score_lexical_dev_by_length.csv'
    __test_length = 'score_lexical_test_by_length.csv'

    def is_valid(self, location: Path):
        """ Verify that all files are present """

        if not (location / self.__dev_length).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_dev_by_length score file!")
        if not (location / self.__test_length).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_test_by_length score file!")
        if not (location / self.__dev_frequency).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_dev_by_frequency score file!")
        if not (location / self.__test_frequency).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_dev_by_frequency score file!")
        if not (location / self.__dev_pairs).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_dev_by_pairs score file!")
        if not (location / self.__test_pairs).is_file():
            raise FileNotFoundError(f"Score folder {location}, is missing lexical_test_by_pairs score file!")

    def __init__(self, location: Path):
        """ Initialise lexical score object """
        self.is_valid(location)
        self.location = location

    @staticmethod
    def _score_invocab(frame):
        """Weighted mean of scores by frequency, excluding OOVs"""
        # filter out OOVs
        frame = frame[frame['frequency'] != 'oov']

        # weighted mean
        return np.average(
            frame['score'].to_numpy(),
            weights=frame['n'].to_numpy())

    def general(self):
        """ Extract general lexical score """
        dev_score = pd.read_csv(self.location / self.__dev_pairs)['score'].mean()
        test_score = pd.read_csv(self.location / self.__test_pairs)['score'].mean()
        # weighted scores
        dev_score_invocab = self._score_invocab(
            pd.read_csv(self.location / self.__dev_frequency)
        )

        test_score_invocab = self._score_invocab(
            pd.read_csv(self.location / self.__test_frequency)
        )

        return {
            'lexical_all': [dev_score, test_score],
            'lexical_invocab': [dev_score_invocab, test_score_invocab]
        }

    def detailed(self):
        """ Extract detailed lexical score """
        frequency_dev = pd.read_csv(self.location / self.__dev_frequency)
        frequency_test = pd.read_csv(self.location / self.__test_frequency)

        by_frequency = pd.merge(frequency_dev, frequency_test,
                                how="outer", on=['frequency'], suffixes=("_dev", "_test"))

        length_dev = pd.read_csv(self.location / self.__dev_length)
        length_test = pd.read_csv(self.location / self.__test_length)

        by_length = pd.merge(length_dev, length_test, how="outer", on=['length'], suffixes=['_dev', '_test'])

        return {
            "by_length": by_length.to_dict(orient='records'),
            "by_frequency": by_frequency.to_dict(orient='records')
        }


class SemanticScore:
    """ Class that extracts lexical scores resume from a scores directory """
    # score files
    __dev_correlation = 'score_semantic_dev_correlation.csv'
    __test_correlation = 'score_semantic_test_correlation.csv'

    def is_valid(self, location: Path):
        """ Verify that all files are present """

        if not (location / self.__dev_correlation):
            raise FileNotFoundError(f"Score folder {location}, is missing semantic_dev_correlation score file!")
        if not (location / self.__test_correlation):
            raise FileNotFoundError(f"Score folder {location}, is missing semantic_test_correlation score file!")

    def __init__(self, location: Path, size: Dict):
        """ Initialise semantic score object """
        self.is_valid(location)
        self.location = location
        self.size = size

    def general(self):
        """ Extract general semantic score """
        dev_correlations = pd.read_csv(self.location / self.__dev_correlation)
        dev_librispeech_mean = dev_correlations[dev_correlations['type'] == 'librispeech']['correlation'].mean()
        dev_synthetic_mean = dev_correlations[dev_correlations['type'] == 'synthetic']['correlation'].mean()

        dev_correlations['size'] = self.size['dev']['size']
        dev_librispeech_wmean = np.average(
            dev_correlations[dev_correlations['type'] == 'librispeech']['correlation'].to_numpy(),
            weights=dev_correlations[dev_correlations['type'] == 'librispeech']['size'].to_numpy())
        dev_synthetic_wmean = np.average(
            dev_correlations[dev_correlations['type'] == 'synthetic']['correlation'].to_numpy(),
            weights=dev_correlations[dev_correlations['type'] == 'synthetic']['size'].to_numpy())

        test_correlations = pd.read_csv(self.location / self.__test_correlation)
        test_librispeech_mean = test_correlations[test_correlations['type'] == 'librispeech']['correlation'].mean()
        test_synthetic_mean = test_correlations[test_correlations['type'] == 'synthetic']['correlation'].mean()

        test_correlations['size'] = self.size['test']['size']
        test_librispeech_wmean = np.average(
            test_correlations[test_correlations['type'] == 'librispeech']['correlation'].to_numpy(),
            weights=test_correlations[test_correlations['type'] == 'librispeech']['size'].to_numpy())
        test_synthetic_wmean = np.average(
            test_correlations[test_correlations['type'] == 'synthetic']['correlation'].to_numpy(),
            weights=test_correlations[test_correlations['type'] == 'synthetic']['size'].to_numpy())

        return {
            "semantic_synthetic": [
                dev_synthetic_mean, test_synthetic_mean],
            "semantic_librispeech": [
                dev_librispeech_mean, test_librispeech_mean],
            "weighted_semantic_synthetic": [
                dev_synthetic_wmean, test_synthetic_wmean],
            "weighted_semantic_librispeech": [
                dev_librispeech_wmean, test_librispeech_wmean]
        }

    def detailed(self):
        """ Extract detailed semantic score """
        dev_correlations = pd.read_csv(self.location / self.__dev_correlation)
        test_correlations = pd.read_csv(self.location / self.__test_correlation)

        ndev_correlations = dev_correlations \
            .set_index(['dataset', dev_correlations.groupby('dataset').cumcount()])['correlation'] \
            .unstack() \
            .reset_index()
        ndev_correlations.columns = ['dataset', 'librispeech', 'synthetic']
        ndev_correlations["set"] = "dev"

        ntest_correlations = test_correlations \
            .set_index(['dataset', test_correlations.groupby('dataset').cumcount()])['correlation'] \
            .unstack() \
            .reset_index()
        ntest_correlations.columns = ['dataset', 'librispeech', 'synthetic']
        ntest_correlations["set"] = "test"

        correlations = ndev_correlations.append(ntest_correlations)
        return correlations.to_dict(orient='records')


class SyntacticScores:
    """ Class that extracts syntactic scores resume from a scores directory """
    # score files
    __dev_pairs = 'score_syntactic_dev_by_pair.csv'
    __test_pairs = 'score_syntactic_test_by_pair.csv'
    __dev_types = 'score_syntactic_dev_by_type.csv'
    __test_types = 'score_syntactic_test_by_type.csv'

    def is_valid(self, location: Path):
        """ Verify that all files are present """

        if not (location / self.__dev_pairs):
            raise FileNotFoundError(f"Score folder {location}, is missing syntactic_dev_by_pair score file!")
        if not (location / self.__test_pairs):
            raise FileNotFoundError(f"Score folder {location}, is missing syntactic_test_by_pair score file!")
        if not (location / self.__dev_types):
            raise FileNotFoundError(f"Score folder {location}, is missing syntactic_dev_by_type score file!")
        if not (location / self.__test_types):
            raise FileNotFoundError(f"Score folder {location}, is missing syntactic_test_by_type score file!")

    def __init__(self, location: Path):
        """ Initialise syntactic score object """
        self.is_valid(location)
        self.location = location

    def general(self):
        """ Extract general semantic score """
        dev_mean = pd.read_csv(self.location / self.__dev_pairs)['score'].mean()
        test_mean = pd.read_csv(self.location / self.__test_pairs)['score'].mean()
        return [dev_mean, test_mean]

    def detailed(self):
        """ Extract detailed semantic score """
        dev_types = pd.read_csv(self.location / self.__dev_types)
        test_types = pd.read_csv(self.location / self.__test_types)

        merged = pd.merge(dev_types, test_types, how="outer", on=["type"], suffixes=("_dev", "_test"))

        return merged.to_dict(orient='records')


class PhoneticScores:
    """ Class that extracts syntactic scores resume from a scores directory """
    # score files
    __scores = 'score_phonetic.csv'

    def is_valid(self, location: Path):
        """ Verify that all files are present """

        if not (location / self.__scores):
            raise FileNotFoundError(f"Score folder {location}, is missing phonetic score file!")

    def __init__(self, location: Path):
        """ Initialise phonetic score object """
        self.is_valid(location)
        self.location = location

    def general(self):
        """ Extract general semantic score """

        def e(d):
            return {s['type']: s['score'] for s in d}

        frame = pd.read_csv(self.location / self.__scores)
        dev_clean = frame[(frame["dataset"] == 'dev') & (frame["sub-dataset"] == 'clean')][['type', 'score']] \
            .to_dict(orient='records')
        dev_other = frame[(frame["dataset"] == 'dev') & (frame["sub-dataset"] == 'other')][['type', 'score']] \
            .to_dict(orient='records')
        test_clean = frame[(frame["dataset"] == 'test') & (frame["sub-dataset"] == 'clean')][['type', 'score']] \
            .to_dict(orient='records')
        test_other = frame[(frame["dataset"] == 'test') & (frame["sub-dataset"] == 'other')][['type', 'score']] \
            .to_dict(orient='records')

        return {
            "phonetic_clean_within": [e(dev_clean)['within'], e(test_clean)['within']],
            "phonetic_clean_across": [e(dev_clean)['across'], e(test_clean)['across']],
            "phonetic_other_within": [e(dev_other)['within'], e(test_other)['within']],
            "phonetic_other_across": [e(dev_other)['across'], e(test_other)['across']]
        }

    @staticmethod
    def detailed():
        """ Extract detailed semantic score """
        # phonetic task has no detailed view of scores
        return {}


@dataclass
class Metadata:
    author: str
    affiliation: str
    description: str
    open_source: bool
    train_set: str
    gpu_budget: float
    visually_grounded: bool
    parameters: Dict
    submission_id: Optional[str] = None
    submission_date: Optional[datetime] = None
    submitted_by: Optional[str] = None

    @staticmethod
    def parse_external_meta(filepath: Path) -> Dict:
        if filepath is None or not filepath.is_file():
            return {}
        elif filepath.suffix == '.json':
            with filepath.open() as fp:
                return json.load(fp)
        else:
            # old txt based file
            submitted_at = None
            with filepath.open() as fp:
                for line in fp.readlines():
                    line = line.rstrip()
                    if line.startswith('submitted-at:'):
                        submitted_at = line.replace('submitted-at:', '').replace(' ', '')
            return {"submitted-at": submitted_at}

    @staticmethod
    def filter_external_meta(data: Dict):
        try:
            sub_data = datetime.fromisoformat(data.get("submitted-at", None))
        except (ValueError, TypeError):
            sub_data = None

        return {
            "submission_date": sub_data,
            "submitted_by": data.get("user", None),
            "submission_id": data.get("submission_id", None)
        }

    @classmethod
    def create_from(cls, filepath: Path, external_meta_file: Path):
        with (filepath / 'meta.yaml').open() as fp:
            meta = yaml.load(fp, Loader=yaml.SafeLoader)

        # parse & filter items of platform metadata
        external_meta = cls.filter_external_meta(cls.parse_external_meta(external_meta_file))

        return cls(**meta, **external_meta)

    def to_dict(self):
        if self.submission_date:
            sub_date = self.submission_date.isoformat()
        else:
            sub_date = datetime.now().isoformat()

        return {
            "submitted_at": sub_date,
            "author": self.author,
            "affiliation": self.affiliation,
            "submitted_by": self.submitted_by,
            "submission_id": self.submission_id,
            "description": self.description,
            "visually_grounded": self.visually_grounded,
            "open_source": self.open_source,
            "train_set": self.train_set,
            "gpu_budget": self.gpu_budget,
            "parameters": self.parameters
        }


class ZeroSpeechSubmission:

    def __init__(self, submission_location: Path, _semantic_size: Dict,
                 score_location: Path, external_meta_file: Path):

        # fetch metadata
        self.description = Metadata.create_from(submission_location, external_meta_file)

        # create scores
        self.lexical = LexicalScores(score_location)
        self.semantic = SemanticScore(score_location, _semantic_size)
        self.syntactic = SyntacticScores(score_location)
        self.phonetic = PhoneticScores(score_location)

    def leaderboard(self):
        """ Build leaderboard object """
        ph = self.phonetic.general()
        le = self.lexical.general()
        se = self.semantic.general()
        sy = self.syntactic.general()
        more = {
            "description": self.description.to_dict(),
            "lexical": self.lexical.detailed(),
            "syntactic": self.syntactic.detailed(),
            "semantic": self.semantic.detailed(),
        }
        return {
            "author_label": self.description.author,
            "set": ['dev', 'test'],
            **le,
            "syntactic": sy,
            **ph,
            **se,
            "more": more
        }


def get_semantic_size(dataset: Path):
    test_size = pd.read_csv(dataset / 'semantic/test/pairs.csv', header=0) \
                  .groupby(['type', 'dataset'], as_index=False).size()
    dev_size = pd.read_csv(dataset / 'semantic/dev/pairs.csv', header=0) \
                 .groupby(['type', 'dataset'], as_index=False).size()
    return {'dev': dev_size, 'test': test_size}
