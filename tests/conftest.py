import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run by verify_equivalence.sh --gpu")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device visible")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
