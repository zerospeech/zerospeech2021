import yaml

from zerospeech2021 import exception


def validate_meta_file(submission_location):
    """ Validation of the meta.yaml in submission

    Testing that the meta.yaml is a valid yaml file and corresponds to the following format:
        author: <str>
        affiliation: <str>
        description: |
          <str>
          model description, may be split on several lines
        open_source: <bool>
        train_set: <str> description of the train set used
        parameters:
          phonetic:
            metric: <str>, must be "cosine", "euclidean", "kl" or "kl_symmetric"
            features_size: <float>, Size (in s) of one feature
          semantic:
            metric: <str>, to be defined
            pooling: <str>, must be "min", "max" or "mean"
    :raises exception.ValidationError for each item not corresponding to prototype.
    """
    meta_file = submission_location / 'meta.yaml'
    if not meta_file.is_file():
        raise exception.ValidationError("missing meta.yaml file")

    with meta_file.open() as fp:
        meta = yaml.safe_load(fp)

    if not isinstance(meta, dict):
        raise exception.ValidationError("meta.yaml file is not valid")

    if not ('author' in meta.keys() and isinstance(meta['author'], str)):
        raise exception.ValidationError(f"meta.yaml: author section not Valid or Missing")

    if not ('affiliation' in meta.keys() and isinstance(meta['affiliation'], str)):
        raise exception.ValidationError(f"meta.yaml: affiliation section not Valid or Missing")

    if not ('description' in meta.keys() and isinstance(meta['description'], str)):
        raise exception.ValidationError(f"meta.yaml: description section not Valid or Missing")

    if not ('open_source' in meta.keys() and isinstance(meta['open_source'], bool)):
        raise exception.ValidationError(f"meta.yaml: open_source section not Valid or Missing")

    if not ('train_set' in meta.keys() and isinstance(meta['train_set'], str)):
        raise exception.ValidationError(f"meta.yaml: train_set section not Valid or Missing")

    if not ('parameters' in meta.keys() and isinstance(meta['parameters'], dict)):
        raise exception.ValidationError(f"meta.yaml: parameters section not Valid or Missing")

    parameters = meta['parameters']

    if not ('phonetic' in parameters.keys() and isinstance(parameters['phonetic'], dict)):
        raise exception.ValidationError(f"meta.yaml: parameters::phonetic section not Valid or Missing")

    phonetic = parameters['phonetic']

    if not ('metric' in phonetic.keys() and phonetic['metric'] in ('cosine', 'euclidean', 'kl', 'kl_symmetric')):
        raise exception.ValidationError(f"meta.yaml: parameters::phonetic::metric section not Valid or Missing")

    if not ('features_size' in phonetic.keys() and isinstance(phonetic['features_size'], float)):
        raise exception.ValidationError(f"meta.yaml: parameters::phonetic::feature_size section not Valid or Missing")

    if not ('semantic' in parameters.keys() and isinstance(parameters['semantic'], dict)):
        raise exception.ValidationError(f"meta.yaml: parameters::semantic section not Valid or Missing")

    semantic = parameters['semantic']

    if not ('metric' in semantic.keys()):
        raise exception.ValidationError(f"meta.yaml: parameters::semantic::metric section not Valid or Missing")

    if not ('pooling' in semantic.keys() and semantic['pooling'] in ('min', 'max', 'mean')):
        raise exception.ValidationError(f"meta.yaml: parameters::phonetic::metric section not Valid or Missing")

