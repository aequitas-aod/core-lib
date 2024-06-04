import aequitas
from aequitas.core.datasets import create_dataset, StandardDataset, DatasetWithBinaryLabelMetrics, DatasetWithRegressionMetrics
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from pathlib import Path
import urllib.request
import aif360

DATA_DIR = Path(aif360.__file__).parent / "data" / "raw"


def _download(url: str, path: Path, timeout=60*5, buffer_size=1024*1024):
    aequitas.logger.warning(f"Downloading {url} to {path}")
    with open(path, 'wb') as file:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if 200 <= response.getcode() < 300:
                while True:
                    buffer = response.read(buffer_size)
                    if not buffer:
                        aequitas.logger.info(f"Successfully downloaded {url} to {path}")
                        return
                    file.write(buffer)
            raise Exception(f"Failed to download {url}: {response.getcode()}")


def _download_if_missing(dirname: str, filename: str, baseurl: str, max_attempts: int = 3):
    dir = DATA_DIR / dirname
    if not dir.exists():
        dir.mkdir()
    file = dir / filename
    if not file.exists():
        url = f"{baseurl}/{filename}"
        while max_attempts > 0:
            try:
                _download(url, file)
                return
            except Exception as e:
                max_attempts -= 1
                if max_attempts == 0:
                    raise e
                else:
                    aequitas.logger.warning(
                        f"Failed to download {url} in {file}. Retrying ({max_attempts} remaining attempts). Cause: {e}"
                    )


def adult(unprivileged_groups=None, privileged_groups=None, **kwargs) -> DatasetWithBinaryLabelMetrics:
    unprivileged_groups = unprivileged_groups or [{'sex': 0}]
    privileged_groups = privileged_groups or [{'sex': 1}]
    for name in ("adult.data", "adult.test", "adult.names"):
        _download_if_missing("adult", name, "https://archive.ics.uci.edu/ml/machine-learning-databases/adult")
    protected_attributes = kwargs.pop('protected_attributes', ['sex'])
    sub_samp = kwargs.pop('sub_samp', None)
    balance = kwargs.pop('balance', None)
    dataset: StandardDataset = load_preproc_data_adult(protected_attributes, sub_samp, balance)
    return create_dataset("binary", unprivileged_groups, privileged_groups, wrap=dataset, **kwargs)
