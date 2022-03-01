import tempfile
from pathlib import Path
from typing import List

import pandas as pd
from Crypto.Hash import MD5
from fsplit.filesplit import Filesplit

from .model import SplitManifest, ManifestFileIndexItem


def md5sum(file_path: Path, chunk_size: int = 8192):
    """ Return a md5 hash of a files content """
    h = MD5.new()

    with file_path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if len(chunk):
                h.update(chunk)
            else:
                break
    return h.hexdigest()


def split_zip_v2(zipfile: Path, chunk_max_size: int = 500000000, hash_parts: bool = True):
    """..."""
    assert zipfile.is_file(), f"entry file ({zipfile}) was not found"
    print(f"splitting {zipfile} into chunks...")

    tmp_loc = Path(tempfile.mkdtemp(dir=f"{zipfile.parents[0]}"))
    fs = Filesplit()
    fs.split(file=f"{zipfile}", split_size=chunk_max_size, output_dir=str(tmp_loc))
    df = pd.read_csv(tmp_loc / 'fs_manifest.csv')
    if hash_parts:
        df['hash'] = df.apply(lambda row: md5sum(
            (tmp_loc / row['filename'])), axis=1)
        index: List[ManifestFileIndexItem] = [ManifestFileIndexItem(file_name=x[0], file_size=x[1], file_hash=x[2])
                                              for x in zip(df['filename'], df['filesize'], df['hash'])]
    else:
        index: List[ManifestFileIndexItem] = [ManifestFileIndexItem(file_name=x[0], file_size=x[1])
                                              for x in zip(df['filename'], df['filesize'])]

    return SplitManifest(
        filename=zipfile.name,
        tmp_location=tmp_loc,
        hash=md5sum(zipfile),
        index=index,
        hashed_parts=hash_parts
    )