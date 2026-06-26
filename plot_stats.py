"""Визуализация emergence rate по дизайнам со статистическими ошибками (интервал Вилсона).

Читает stats.json (артефакт collect_stats.py) и строит столбчатую диаграмму: высота столбца —
доля прогонов с возникновением института репутации, асимметричные усы — 95% интервал Вилсона.
Сохраняет PNG.

    uv run python plot_stats.py [--in stats.json] [--out stats.png] [--title TEXT]
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")            # рендер в файл без дисплея
import matplotlib.pyplot as plt  # noqa: E402 (после use("Agg"))


def load_designs(path: str) -> list[dict]:
    """Прочитать список дизайнов из stats.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("designs", [])


def render(designs: list[dict], out_path: str,
           title: str = "Emergence of reputation by design") -> None:
    """Построить bar chart с асимметричными усами интервала Вилсона и сохранить PNG.

    Raises:
        ValueError: пустой список дизайнов — рисовать нечего.
    """
    if not designs:
        raise ValueError("в stats.json нет дизайнов — нечего рисовать")
    labels = [d["name"] or d["config_hash"][:8] for d in designs]
    rates = [d["rate"] for d in designs]
    lo_err = [d["rate"] - d["ci_lo"] for d in designs]
    hi_err = [d["ci_hi"] - d["rate"] for d in designs]
    x = range(len(designs))

    fig, ax = plt.subplots(figsize=(max(6.0, len(designs) * 1.2), 5.0))
    ax.bar(x, rates, yerr=[lo_err, hi_err], capsize=6, color="#4878a8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля прогонов с возникновением института")
    ax.set_title(title)
    for i, d in enumerate(designs):                 # n под каждым столбцом
        ax.text(i, 0.02, f"n={d['n']}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _flag(args: list[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    args = sys.argv[1:]
    in_path = _flag(args, "--in", "stats.json")
    out_path = _flag(args, "--out", "stats.png")
    title = _flag(args, "--title", "Emergence of reputation by design")
    render(load_designs(in_path), out_path, title)
    print(f"Записано: {out_path}")


if __name__ == "__main__":
    main()
