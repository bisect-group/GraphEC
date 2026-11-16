import resource
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.utils import raise_open_file_limit, set_tf32  # noqa: E402


def test_soft_open_file_limit_rises_to_the_hard_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        raise_open_file_limit()
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == (hard, hard)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_tf32_switch_sets_matmul_and_cudnn() -> None:
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        set_tf32(True)
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        set_tf32(False)
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
