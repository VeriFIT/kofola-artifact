#!/usr/bin/env python3
"""
Script to display experimental results for complement benchmarks.
Shows terminal tables and generates plots based on evaluation data.
"""

import sys
import os
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate

# Import evaluation functions
import eval_functions
importlib.reload(eval_functions)
from eval_functions import *

# Configuration
KOFOLA_VERSION = "7bbf4a1"
KOFOLA_TACAS23 = "kofola-tacas23-tacas23"
KOFOLA_SUBS_TUP = "kofola-subs-tup-7bbf4a1"
SPOT = "spot-2.14.2"
KOFOLA = "kofola-" + KOFOLA_VERSION
RANKER = "ranker-ranker"

# Timeout (in seconds)
TIMEOUT = 120

# Tools list - avoiding duplicates while maintaining order
TOOLS = list(dict.fromkeys([
    KOFOLA,
    SPOT,
    KOFOLA_TACAS23,
    KOFOLA_SUBS_TUP,
    RANKER
]))

TOOLS_CHECK = []

BENCHES = [
    "advanced_automata_termination",
    "autohyper",
    "pecan",
    "s1s",
    "state_of_buchi",
    "seminator",
    "ldba4ltl",
]

BENCHMARK_TO_LATEX = {
    "seminator": "Seminator",
    "advanced_automata_termination": "Termination",
    "autohyper": "AutoHyper",
    "pecan": "Pecan",
    "s1s": "S1S",
    "state_of_buchi": "State-of-Buchi",
    "ldba4ltl": "LDBA4LTL"
}

TOOLS_PAPER = list(dict.fromkeys([
    KOFOLA,
    SPOT,
    KOFOLA_TACAS23,
    KOFOLA_SUBS_TUP,
    RANKER
]))

TOOLS_PLOT = list(dict.fromkeys([
    KOFOLA,
    SPOT,
    KOFOLA_TACAS23,
    KOFOLA_SUBS_TUP,
    RANKER,
]))

TOOL_MAP = {
    KOFOLA: "Kofola",
    SPOT: "Spot",
    KOFOLA_TACAS23: "KofolaOld",
    KOFOLA_SUBS_TUP: "KofolaSlice",
    RANKER: "Ranker"
}


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def main():
    """Main function to load data and display results."""
    
    df_all = load_benches(BENCHES, TOOLS + TOOLS_CHECK, TIMEOUT)
    
    # Build and display statistics for LaTeX
    print_section_header("Complement states statistics")

    df_stats = build_complement_stats_df(df_all, TOOLS_PAPER, BENCHES)
    cnt = count_unsupported_instances(BENCHES, TOOLS_PAPER, TIMEOUT)
    
    unsupported = {k: sum(v.values()) for k, v in cnt.items()}
    df_unsupported = pd.DataFrame.from_dict(
        unsupported, orient="index", columns=["unsupported"]
    ).reset_index().rename(columns={"index": "tool"})
    
    # Join unsupported counts into df_stats and update OOR
    df_stats = df_stats.merge(df_unsupported[['tool', 'unsupported']], on='tool', how='left')
    df_stats['unsupported'] = df_stats['unsupported'].fillna(0).astype(int)
    
    # Subtract unsupported from OOR and ensure non-negative integer
    df_stats['OOR'] = (df_stats['OOR'] - df_stats['unsupported']).clip(lower=0).astype(int)
    
    # Compute total number of instances in the requested benches
    total_instances = df_all[df_all['benchmark'].isin(BENCHES)]['benchmark'].nunique()
    # In case benchmarks have multiple instances per benchmark, better compute unique names per benchmark
    # but here we interpret 'all' as number of unique (benchmark,name) pairs across chosen benches
    total_instances = df_all[df_all['benchmark'].isin(BENCHES)].drop_duplicates(subset=['benchmark', 'name']).shape[0]

    # Add 'all' column and compute 'unsolved' = all - solved - unsupported (clipped at 0)
    df_stats['all'] = int(total_instances)
    df_stats['unsolved'] = (df_stats['all'] - df_stats['solved'] - df_stats['unsupported']).clip(lower=0).astype(int)
    
    # Display statistics table
    # Map internal tool identifiers to display names according to TOOL_MAP for readability
    df_stats_display = df_stats.copy()
    df_stats_display['tool'] = df_stats_display['tool'].map(TOOL_MAP).fillna(df_stats_display['tool'])
    # Keep only requested columns for display
    cols_to_show = ['tool', 'unsolved', 'avg', 'unsupported']
    df_stats_display = df_stats_display.loc[:, [c for c in cols_to_show if c in df_stats_display.columns]]

    # Pretty-print the filtered DataFrame to terminal using tabulate
    print("Statistics DataFrame:")
    headers = list(df_stats_display.columns)
    table = tabulate(df_stats_display.values, headers=headers, tablefmt="fancy_grid", numalign="right", stralign="left")
    print(table)
    print()
    
    # Generate plots
    print_section_header("Generating Plots")
    
    TOOL_FOR_COMPARISON = KOFOLA
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    
    print(f"Generating scatter plots comparing {TOOL_FOR_COMPARISON} with other tools...")
    print(f"Plots will be saved in: {plot_dir.absolute()}\n")
    
    for tool in TOOLS_PLOT:
        if tool != TOOL_FOR_COMPARISON:
            file_name = f"compl_{TOOL_FOR_COMPARISON}_{tool}"
            print(f"  Generating plot: {file_name}.png")
            try:
                plot = scatter_plot_states(
                    df_all,
                    TOOL_FOR_COMPARISON,
                    tool,
                    color_column="benchmark",
                    legend_name_map=BENCHMARK_TO_LATEX,
                    tool_name_map=TOOL_MAP,
                    show_legend=False,
                    file_name_to_save=file_name
                )
                # The plot is saved by the function itself
            except Exception as e:
                print(f"    Error generating plot: {e}")
    
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
