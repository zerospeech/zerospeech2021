""" Script to download and setup the file for the Zerospeech 2021 challenge """
import hashlib
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import click
import pandas
import requests
from tqdm import tqdm


ROOT_URL = 'https://download.zerospeech.com/2021/'
URLS = {
    'md5': ROOT_URL + 'md5checksum.txt',
    'dataset': ROOT_URL + 'dataset.zip',
    'submission_baseline': ROOT_URL + 'baseline_submission.zip',
    'submission_random': ROOT_URL + 'random_submission.zip'
}


def download_file(url, location: Path, filename=None):
    """Download a file from a url in a specific location

    :arg url : the url of the file to download
    :arg location : the location in which to save the file
    :arg filename : the name of the file (to keep same name as in URL leave
      value as None)
    :raises ValueError if dowload failed
    :note writes downloaded file in stream mode to handle large files

    """
    if not filename:
        filename = url.split('/')[-1]
    req = requests.get(url, stream=True, allow_redirects=True)

    if not req.ok:
        raise ValueError(f'failed to retrieve {url}')

    with (location / filename).open('wb') as fh:
        for chunk in tqdm(req.iter_content(chunk_size=1024),
                          desc=f'downloading {filename} '):
            if chunk:  # filter out keep-alive new chunks
                fh.write(chunk)
                fh.flush()
    return location / filename


def checksum(
        filename: Path, hash_fn=hashlib.md5, chunk_num_blocks=128, hexa=True):
    """ Do a checksum of a large file

    :arg filename : the path of the file to checksum
    :arg hash_fn : the function to use for checksum (default md5)
    :arg chunk_num_blocks : the size of each chunk (md5 default 128)
    :arg hexa : return string with hex version instead of bytes
    :note checksum is computed by iterating over chunks of the file to allow
    checksums of files that too large to load them into memory.
    """
    fn = hash_fn()
    with filename.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_num_blocks*fn.block_size), b''):
            fn.update(chunk)
    if hexa:
        return fn.hexdigest()
    return fn.digest()


def load_md5():
    """ Loads list of reference md5s from the server

    :return: a dict containing { [filename]: md5_checksum }
    :raises: ValueError if download failed
    """
    print("Downloading checksum files from server...")
    md5_file = Path(tempfile.mktemp())

    main = requests.get(URLS['md5'], allow_redirects=True)
    if not main.ok:
        raise ValueError(f'failed to retrieve {URLS["md5"]}')

    with md5_file.open('wb') as fh:
        fh.write(main.content)

    md5_items = pandas.read_csv(md5_file, sep=" ", header=None)
    md5_items.columns = ["md5", 'empty', 'filename']
    del md5_items['empty']

    return dict(zip(md5_items.filename, md5_items.md5))


def extract_archive(archive_location: Path):
    """ Extracts an archive from path

    :arg archive_location : the location of the archive
    :format .zip .tar.gz
    :raises FileNotFoundError if the archive file does not exist
    :raises TypeError if the archive format is unknown

    """
    print(f"> extracting {archive_location.name}")
    if not archive_location.is_file():
        raise FileNotFoundError(f'Archive: {archive_location} does not exist!')

    extract_location = archive_location.parents[0]
    archive_format = ''.join(archive_location.suffixes)
    if archive_format in ('.tar.gz', '.tgz'):
        # extract tar.gz
        with tarfile.open(archive_location, "r:gz") as tar_ref:
            tar_ref.extractall(extract_location)
    elif archive_format == '.zip':
        # extract zip
        with zipfile.ZipFile(archive_location, 'r') as zip_ref:
            zip_ref.extractall(extract_location)
    else:
        raise TypeError(f"Archive format {archive_format} is not known!")


def check_archive(archive_file: Path, md5_dict: Dict):
    """ Verify an archive against the reference md5_checksum

    :arg archive_file: the archive filename
    :arg md5_dict: a dictionary of the reference checksums
    :raises ValueError if checksums don't match
    """
    print(f"> verifying archive {archive_file.name} integrity")
    original_md5 = md5_dict.get(archive_file.name)
    md5sum = checksum(archive_file)
    if not original_md5 == md5sum:
        raise ValueError(
            f"archive : {archive_file} does not validate checksums")


def download_dataset(location: Path, md5_dict: Dict):
    """ Downloads and extracts the dataset archive of the zerospeech 2021 dataset

    :param location: location to download the files
    :param md5_dict: dictionary of md5 checksums
    """
    print("Downloading challenge dataset...")
    dataset_archive = download_file(URLS['dataset'], location)
    # check archive integrity
    check_archive(dataset_archive, md5_dict)
    # extract archive
    extract_archive(dataset_archive)
    # delete archive
    dataset_archive.unlink()


def download_submission_baseline(location: Path, md5_dict: Dict):
    """ Downloads and extracts the baseline submission of the zerospeech 2021

    :param location: location to download the files
    :param md5_dict: dictionary of md5 checksums
    """
    print("Downloading baseline submission...")
    baseline_archive = download_file(URLS['submission_baseline'], location)
    # check archive integrity
    check_archive(baseline_archive, md5_dict)
    # extract archive
    extract_archive(baseline_archive)
    # delete archive
    baseline_archive.unlink()


def download_submission_random(location: Path, md5_dict: Dict):
    """ Downloads and extracts the random submission of the zerospeech 2021

    :param location: location to download the files
    :param md5_dict: dictionary of md5 checksums
    """
    print("Downloading random submission...")
    submission = download_file(URLS['submission_random'], location)
    # check archive integrity
    check_archive(submission, md5_dict)
    # extract archive
    extract_archive(submission)
    # delete archive
    submission.unlink()


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('output_directory', type=Path)
@click.option('--no-submission', '-S', is_flag=True,
              help='download only the dataset, not the sample submissions')
def download(output_directory, no_submission):
    """Download datasets for the Zero Resource Speech Challenge 2021

    OUTPUT_DIRECTORY is the directory where to store the downloaded content.

    The following sub-directories are created: "dataset" contains the test and
    dev datasets for the 4 tasks, "baseline_submission" contains the complete
    submission for the challenge baseline, "random_submission" contains a
    random exemple of submission.

    """
    try:
        md5_values = load_md5()

        location = Path(output_directory)
        location.mkdir(exist_ok=True, parents=True)

        download_dataset(location, md5_values)

        if not no_submission:
            download_submission_baseline(location, md5_values)
            download_submission_random(location, md5_values)
    except (ValueError, TypeError, FileNotFoundError) as error:
        print(f'ERROR: {error}')
