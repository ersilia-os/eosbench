"""Tests for the catalog export used by the GitHub Pages site (scripts/export_catalog.py).

Runs fully offline: a fake ``head_check`` stands in for the S3 availability probe.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import export_catalog as ec  # noqa: E402


def test_build_catalog_shape_and_availability():
    # Fake checker: only "ames" datasets are "available". No network.
    payload = ec.build_catalog(head_check=lambda url: "/ames/" in url)

    assert payload["s3_base"] == ec.S3_BASE
    assert payload["generated_at"].endswith("+00:00")
    datasets = payload["datasets"]
    assert payload["n_datasets"] == len(datasets)
    assert payload["n_available"] == sum(1 for d in datasets if d["available"])

    # Every record carries the full union of keys plus the derived fields.
    required = set(ec._UNION_KEYS) | {"columns", "data_url", "fetch_cmd", "available"}
    for d in datasets:
        assert required <= set(d), f"{d['name']} missing keys: {required - set(d)}"
        assert isinstance(d["columns"], list) and d["columns"]
        assert d["data_url"].startswith(ec.S3_BASE)
        assert d["fetch_cmd"] == (
            f"eosbench fetch --source {d['source']} --task {d['task']} --dataset {d['name']}"
        )

    # Availability reflects the injected checker.
    ames = [d for d in datasets if d["name"] == "ames"]
    assert ames and all(d["available"] for d in ames)
    assert all(not d["available"] for d in datasets if d["name"] != "ames")


def test_build_catalog_is_strict_json_serializable():
    # The browser uses JSON.parse, which rejects NaN/Infinity — _clean must produce valid JSON.
    payload = ec.build_catalog(head_check=lambda url: False)
    text = json.dumps(payload, allow_nan=False)  # raises if any NaN/inf leaked through
    assert json.loads(text)["n_datasets"] == payload["n_datasets"]


def test_known_sources_present():
    payload = ec.build_catalog(head_check=lambda url: False)
    sources = {d["source"] for d in payload["datasets"]}
    assert {"tdcommons", "moleculenet"} <= sources
