from aequitas.core.datasets import create_dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
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


def adult(unprivileged_groups, privileged_groups, **kwargs):
    for name in ("adult.data", "adult.test", "adult.names"):
        _download_if_missing("adult", name, "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/")
    dataset = load_preproc_data_adult()
    return create_dataset("binary", unprivileged_groups, privileged_groups, wrap=dataset)

