from __future__ import annotations

import pytest

import plot_stats


def _designs():
    return [
        {"config_hash": "d1", "name": "base", "n": 10, "n_emerged": 7,
         "rate": 0.7, "ci_lo": 0.4, "ci_hi": 0.9},
        {"config_hash": "d2", "name": None, "n": 5, "n_emerged": 1,
         "rate": 0.2, "ci_lo": 0.04, "ci_hi": 0.62},
    ]


def test_render_writes_nonempty_png(tmp_path):
    out = tmp_path / "p.png"
    plot_stats.render(_designs(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        plot_stats.render([], str(tmp_path / "p.png"))


def test_load_designs_reads_artifact(tmp_path):
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"filters": {}, "designs": _designs()}))
    assert len(plot_stats.load_designs(str(p))) == 2
