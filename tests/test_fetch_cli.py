import pytest

from eosbench.dataset import get_catalog
from eosbench.cli.fetch import _select_datasets, _fetch_many
from eosbench.cli import fetch as fetch_module


def test_select_datasets_filters_by_source_and_task():
    targets = _select_datasets("tdcommons", "classification", None)
    assert len(targets) == len(get_catalog(source="tdcommons", task="classification"))
    assert all(source == "tdcommons" and task == "classification" for source, _, task in targets)


def test_select_datasets_name_filter_matches_across_sources():
    # "ames" is prepared under both tdcommons (PyTDC) and polaris (Polaris's official
    # split) -- a name filter with no --source matches the family in every source.
    targets = _select_datasets(None, "classification", "ames")
    assert sorted(targets) == [
        ("polaris", "ames", "classification"),
        ("tdcommons", "ames", "classification"),
    ]


def test_select_datasets_no_source_covers_every_source():
    all_sources = {source for source, _, _ in _select_datasets(None, "classification", None)}
    assert all_sources == {"tdcommons", "moleculenet", "polaris"}


def test_fetch_many_continues_past_a_failure(monkeypatch):
    calls = []

    def fake_mirror_dataset(*, source, dataset, featurization, output_dir, task, from_dir):
        calls.append(dataset)
        if dataset == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(fetch_module, "mirror_dataset", fake_mirror_dataset)

    targets = [("src", "good1", "classification"), ("src", "bad", "classification"), ("src", "good2", "classification")]
    failures = _fetch_many(targets, featurization="morgan", output_dir=".", from_dir=None)

    assert calls == ["good1", "bad", "good2"]
    assert failures == ["bad"]
