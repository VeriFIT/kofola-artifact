#!/usr/bin/env python3
"""
Script to display experimental results for inclusion benchmarks.
Shows terminal tables and generates plots based on evaluation data.

This mirrors the complement-case script but uses the inclusion notebook's
LaTeX Tables logic (without emitting LaTeX), and saves comparison plots.
"""

import sys
import os
import importlib
from pathlib import Path

import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

# Import evaluation functions
import eval_functions
importlib.reload(eval_functions)
from eval_functions import *  # noqa: F401,F403

# -----------------------------------------------------------------------------
# Configuration (mirrors eval_incl.ipynb "LaTeX Tables" section inputs)
# -----------------------------------------------------------------------------

# Tool identifiers used in precomputed results
KOFOLA = "kofola-7bbf4a1"
SPOT = "spot-2.14.2"
SPOT_FORQ = "spot-forq-2.14.2"

FORKLIFT = "forklift-forklift"
RABIT = "rabit-2.5.1"
BAIT = "bait-bait"

# Timeout (in seconds)
TIMEOUT = 120

# Tool sets (preserve order, avoid duplicates)
TOOLS = list(dict.fromkeys([
    KOFOLA,
    SPOT,
    SPOT_FORQ,
]))

BA_TOOLS = list(dict.fromkeys([
    FORKLIFT,
    RABIT,
    BAIT
]))

# Benchmarks used for inclusion evaluation
BENCHES = [
    "autohyper",
    "rabit",
    "termination",
    "pecan",
]

# Mapping for nice benchmark names
BENCHMARK_TO_LATEX = {
    "autohyper": "AutoHyper",
    "termination": "Termination",
    "pecan": "Pecan",
    "rabit": "Concurrent",
}

# Full set of tools to present in paper-style comparisons
TOOLS_PAPER = list(dict.fromkeys([
    KOFOLA,
    SPOT,
    SPOT_FORQ,
    FORKLIFT,
    RABIT,
    BAIT,
]))

# Human-friendly tool names for tables/plots
TOOL_MAP = {
    KOFOLA: "Kofola",
    SPOT: "Spot(Det)",
    SPOT_FORQ: "Spot(FORQ)",
    FORKLIFT: "Forklift",
    RABIT: "Rabit",
    BAIT: "Bait",
}

# The primary tool used for pairwise comparisons in plots
TOOL_FOR_COMPARISON = KOFOLA


def print_section_header(title: str) -> None:
    """Print a formatted section header to the terminal."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def main() -> None:
    """Load inclusion data, print stats, and generate comparison plots."""

    # Load non-BA tools (suppress warnings coming from the loader)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_all_non_ba = load_benches_incl(BENCHES, TOOLS, TIMEOUT)

    # Load BA tools and convert to HOA-compatible format, then combine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_ba = load_benches_incl(BENCHES, BA_TOOLS, TIMEOUT)
    df_ba_hoa = ba_bench_to_hoa(df_ba)

    # Combine results by joining on name and benchmark
    df_all = left_join_on_name_benchmark(df_all_non_ba, df_ba_hoa)

    print_section_header("Inclusion statistics")

    # Build stats DataFrame (omit inclusion_stats_to_latex; we print a terminal table)
    df_stats = build_inclusion_stats_df(
        df_all,
        TOOLS_PAPER,
        BENCHES,
        tool_for_comparison=TOOL_FOR_COMPARISON,
    )

    # Map internal tool identifiers to display names for readability (if column present)
    df_stats_display = df_stats.copy()
    if "tool" in df_stats_display.columns:
        df_stats_display["tool"] = (
            df_stats_display["tool"].map(TOOL_MAP).fillna(df_stats_display["tool"])
        )

    # Rename OOR -> unsolved and drop avg/med columns if present
    if "OOR" in df_stats_display.columns:
        df_stats_display = df_stats_display.rename(columns={"OOR": "unsolved"})
    for _col in ["avg", "med"]:
        if _col in df_stats_display.columns:
            df_stats_display = df_stats_display.drop(columns=[_col])

    # Pretty-print the DataFrame to terminal using tabulate
    print("Statistics DataFrame:")
    headers = list(df_stats_display.columns)
    table = tabulate(
        df_stats_display.values,
        headers=headers,
        tablefmt="fancy_grid",
        numalign="right",
        stralign="left",
        floatfmt=".3f",
    )
    print(table)
    print()

    # ------------------------------------------------------------------
    # Plots: pairwise comparisons vs TOOL_FOR_COMPARISON
    # ------------------------------------------------------------------
    print_section_header("Generating Plots")

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    print(
        f"Generating scatter plots comparing {TOOL_MAP.get(TOOL_FOR_COMPARISON, TOOL_FOR_COMPARISON)} with other tools..."
    )
    print(f"Plots will be saved in: {plot_dir.absolute()}\n")

    for tool in TOOLS_PAPER:
        if tool == TOOL_FOR_COMPARISON:
            continue
        file_name = f"incl_{TOOL_FOR_COMPARISON}_{tool}"
        print(f"  Generating plot: {file_name}.png")
        try:
            # The plotting function saves the file when file_name_to_save is provided
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _ = scatter_plot(
                    df_all,
                    TOOL_FOR_COMPARISON,
                    tool,
                    color_column="benchmark",
                    timeout=TIMEOUT,
                    legend_name_map=BENCHMARK_TO_LATEX,
                    tool_name_map=TOOL_MAP,
                    show_legend=False,
                    file_name_to_save=file_name,
                )
        except Exception as e:
            print(f"    Error generating plot for {tool}: {e}")

    print("\nPlot generation completed!")

    print_section_header("Evaluation Complete")
    print("All results have been displayed and plots have been generated.")


if __name__ == "__main__":
    # Change to the eval directory if not already there
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
