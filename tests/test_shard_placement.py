import torch

from emulator_bench import cache_features


def test_shard_positions():
    assert cache_features.shard_positions(2, 2) == [0, 1, 0, 1]


def test_shard_worker_sets_current_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    items = [{"cache_key": "a"}]
    assert cache_features._shard_worker(lambda x, **k: [x[0]["cache_key"]], 1, items, {}) == ["a"]
    assert cache_features._shard_worker(lambda x, **k: [], None, items, {}) == []
    assert calls == [1]


def test_shard_worker_caps_gpu_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda p: calls.append(("device", p)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: type("P", (), {"total_memory": 20 * 1024**3}),
    )
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f))
    run = lambda x, **k: []  # noqa: E731
    cache_features._shard_worker(run, 1, [], {}, 5)
    cache_features._shard_worker(run, 1, [], {}, 50)
    cache_features._shard_worker(run, 1, [], {})
    assert calls == [("device", 1), 0.25, ("device", 1), 1.0, ("device", 1)]


def test_gpu_memory_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--gpu-memory-gb", "6.5"])
    assert cache_features.parse_args().gpu_memory_gb == 6.5
    monkeypatch.setattr("sys.argv", ["x"])
    assert cache_features.parse_args().gpu_memory_gb is None
