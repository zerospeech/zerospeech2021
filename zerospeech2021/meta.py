"""Validation of meta.yaml"""

import numpy as np
import scipy.spatial
import yaml

from zerospeech2021.exception import ValidationError, MismatchError


def _validate_entries(meta, entries, prefix=None):
    if sorted(meta.keys()) != sorted(entries.keys()):
        message = 'invalid entries'
        if prefix:
            message += f' in {prefix}'
        raise MismatchError(message, entries.keys(), meta.keys())

    for key, value in entries.items():
        _validate_entry(meta, key, value[0], values=value[1], prefix=prefix)


def _validate_entry(meta, name, expected_type, values=None, prefix=None):
    prefix = prefix + '/' if prefix else ''

    if name not in meta:
        raise ValidationError(f'{prefix}{name} section missing')

    value = meta[name]
    if not isinstance(value, expected_type):
        raise ValidationError(
            f'{prefix}{name} must be a {expected_type}, it is {type(value)}')

    if values and value not in values:
        raise ValidationError(
            f'{prefix}{name} must be in ({", ".join(values)}) but is {value}')

    if expected_type == str and not value:
        raise ValidationError(f'{prefix}{name} must not be an empty string')


def _validate_scipy_metric(metric):
    """"Raises a ValidationError if `metric` is not a valid metric in scipy"""
    try:
        scipy.spatial.distance.cdist(
            np.ones((5, 2)), np.ones((5, 2)), metric)
    except:
        raise ValidationError(f'invalid metric for semantic: {metric}')


def validate(submission):
    """Validation of the meta.yaml in submission

    Testing that submission/meta.yaml is a valid yaml file and corresponds to
    the following format:

        author: <str>
        affiliation: <str>
        description: <str>
        open_source: <bool>
        train_set: <str>
        budget: <float>
        parameters:
          phonetic:
            metric: <str>, "cosine", "euclidean", "kl" or "kl_symmetric"
            frame_shift: <float>
          semantic:
            metric: <str>
            pooling: <str>, "min", "max" or "mean"

    Raises
    ------
    exception.ValidationError
        For any item not corresponding to prototype.

    """
    meta_file = submission / 'meta.yaml'

    if not meta_file.is_file():
        raise ValidationError("missing meta.yaml file")

    try:
        meta = yaml.safe_load(meta_file.open('r').read().replace('\t', ' '))
    except yaml.YAMLError as err:
        raise ValidationError(f'failed to parse {meta_file}: {err}')

    if not meta or not isinstance(meta, dict):
        raise ValidationError("meta.yaml file is not valid")

    # top level entries
    _validate_entries(
        meta,
        {'author': (str, None),
         'affiliation': (str, None),
         'description': (str, None),
         'open_source': (bool, None),
         'train_set': (str, None),
         'budget': (float, None),
         'parameters': (dict, None)})

    # parameters entries
    _validate_entries(
        meta['parameters'],
        {'phonetic': (dict, None), 'semantic': (dict, None)},
        prefix='parameters')

    # parameters/phonetic level
    _validate_entries(
        meta['parameters']['phonetic'],
        {'metric': (str, ['cosine', 'euclidean', 'kl', 'kl_symmetric']),
         'frame_shift': (float, None)},
        prefix='parameters/phonetic')

    # parameters/semantic level
    _validate_entries(
        meta['parameters']['semantic'],
        {'metric': (str, None),
         'pooling': (str, ['min', 'max', 'mean'])},
        prefix='parameters/semantic')

    _validate_scipy_metric(meta['parameters']['semantic']['metric'])

    return meta['open_source']
