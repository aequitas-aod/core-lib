from aequitas.core.datasets import create_dataset
from .adult_dataset import AdultDataset
from pathlib import Path
import urllib.request
import aif360

DATA_DIR = Path(aif360.__file__).parent / "data" / "raw"


def _download_if_missing(dirname: str, filename: str, baseurl: str):
    dir = DATA_DIR / dirname
    if not dir.exists():
        dir.mkdir()
    file = dir / filename
    if not file.exists():
        url = f"{baseurl}/{filename}"
        urllib.request.urlretrieve(url, file)


def adult(**kwargs):
    for name in ("adult.data", "adult.test", "adult.names"):
        _download_if_missing("adult", name, "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/")
    return create_dataset("adult", **kwargs)
