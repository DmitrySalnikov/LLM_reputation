from __future__ import annotations

import json
import sqlite3

import pytest
from conftest import add_run

import collect_stats
from src.stats.aggregate import DesignStat
from src.stats.selection import RunFilter
from src.storage.schema import init_schema


def test_collect_groups_by_design_from_file_db(tmp_path):
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    init_schema(c)
    add_run(c, 1, config_hash="d1", emerged=True)
    add_run(c, 2, config_hash="d1", emerged=False)
    add_run(c, 3, config_hash="d2", emerged=True)
    c.close()

    stats = collect_stats.collect(str(db), RunFilter())
    by_hash = {s.config_hash: s for s in stats}
    assert by_hash["d1"].n == 2 and by_hash["d1"].n_emerged == 1
    assert by_hash["d2"].rate == pytest.approx(1.0)


def test_stats_to_json_shape():
    stats = [DesignStat("d1", "base", 2, 1, 0.5, 0.24, 0.76, (1, 2))]
    obj = collect_stats.stats_to_json(stats, RunFilter(include_designs=("d1",)))
    assert obj["filters"]["include_designs"] == ["d1"]
    d = obj["designs"][0]
    assert d["config_hash"] == "d1" and d["run_ids"] == [1, 2]


def test_write_json_and_csv_roundtrip(tmp_path):
    stats = [DesignStat("d1", "base", 2, 1, 0.5, 0.24, 0.76, (1, 2))]
    jp = tmp_path / "s.json"
    cp = tmp_path / "s.csv"
    collect_stats.write_json(str(jp), collect_stats.stats_to_json(stats, RunFilter()))
    collect_stats.write_csv(str(cp), stats)
    assert json.loads(jp.read_text())["designs"][0]["n"] == 2
    csv_text = cp.read_text()
    assert "config_hash,name,n,n_emerged,rate,ci_lo,ci_hi" in csv_text
    assert "d1,base,2,1,0.5000,0.2400,0.7600" in csv_text
