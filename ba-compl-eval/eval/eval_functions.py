import pandas as pd
import json
import numpy as np
import mizani.formatters as mizani
import plotnine as p9
import tabulate as tab
from argparse import Namespace
import io
import os
import sys
from pathlib import Path
from enum import Enum

import pyco_proc

def read_latest_result_file(bench, method, tool, timeout):
    assert tool != ""

    #substring to filter files with the same timeout
    timeout_str = f"to{timeout}-"
    matching_files = []
    for root, _, files in os.walk(method + "/" + bench):
        for file in files:
            if tool in file and timeout_str in file:
                matching_files.append(os.path.join(root, file))
    if not matching_files:
        return ""
    latest_file_name = sorted(matching_files, key = lambda x: x[-23:])[-1]
    with open(latest_file_name) as latest_file:
        return latest_file.read()


def count_unsupported_instances(benches, tools, timeout=120):
    """Count unsupported instances for given benchmarks and tools.
    
    For each tool and benchmark, this function reads the latest result file
    and searches for error patterns indicating unsupported automaton types.
    The following error patterns are counted:
    - "Unsupported automaton type"
    - "cola requires Buchi condition on input"
    
    Args:
        benches (list): List of benchmark names to analyze
        tools (list): List of tool names to analyze
        timeout (int, optional): Timeout value for filtering result files. Defaults to 120.
    
    Returns:
        dict: A nested dictionary with structure:
              {tool: {benchmark: count}}
              where count is the number of unsupported instances found.
    
    Example:
        >>> results = count_unsupported_instances(['pecan', 's1s'], ['cola-cola', 'ranker-ranker'])
        >>> print(results)
        {'cola-cola': {'pecan': 150, 's1s': 0}, 'ranker-ranker': {'pecan': 0, 's1s': 5}}
    """
    import re
    
    # Define error patterns to search for
    error_patterns = [
        r"Unsupported automaton type",
        r"cola requires Buchi condition on input"
    ]
    
    # Compile patterns for efficient searching
    compiled_patterns = [re.compile(pattern) for pattern in error_patterns]
    
    results = {}
    
    for tool in tools:
        results[tool] = {}
        for bench in benches:
            # Read the result file content
            content = read_latest_result_file(bench, "compl", tool, timeout)
            
            if not content:
                results[tool][bench] = 0
                continue
            
            # Count occurrences of each error pattern
            total_count = 0
            for pattern in compiled_patterns:
                matches = pattern.findall(content)
                total_count += len(matches)
            
            results[tool][bench] = total_count
    
    return results


def load_benches(benches, tools, timeout = 120):
    dfs = dict()
    for bench in benches:
        input_data = ""
        for tool in tools:
            assert tool != ""
            input_data += read_latest_result_file(bench, "compl", tool, timeout)
        # pyco_proc.proc_res writes CSV to stdout; capture that output into a buffer
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            pyco_proc.proc_res(io.StringIO(input_data), Namespace(csv=True, html=False, text=False, tick=False))
        finally:
            sys.stdout = old_stdout
        csv_output = buf.getvalue()
        df = pd.read_csv(
                io.StringIO(csv_output),
                sep=";",
                dtype='unicode',
        )
        # for the case one tool has only timeouts
        for tool in tools:
            if tool+"-states" not in df.keys():
                df[tool+"-states"] = "TO"
      
        df["benchmark"] = bench
        dfs[bench] = df

    # Collect all available columns to include check columns if present
    all_dfs = pd.concat(dfs, ignore_index=True)
    base_columns = ["benchmark", "name"]
    tool_columns = []
    
    for tool in tools:
        tool_columns.extend([f"{tool}-states", f"{tool}-runtime"])
        # Include check column if it exists
        if f"{tool}-check" in all_dfs.columns:
            tool_columns.append(f"{tool}-check")
    
    df_runtime_result = all_dfs[base_columns + tool_columns]
    
    for tool in tools:
        states_ser = pd.to_numeric(df_runtime_result[f"{tool}-states"], errors='coerce')
        mask_non_numeric = states_ser.isna()
        df_runtime_result.loc[mask_non_numeric, f"{tool}-runtime"] = float(timeout)
        # runtime columns should be floats
        df_runtime_result[f"{tool}-runtime"] = pd.to_numeric(df_runtime_result[f"{tool}-runtime"], errors='coerce').astype(float)

    df_all = df_runtime_result #.merge(df_stats)
    return df_all

def load_benches_incl(benches, tools, timeout=120):
    """Load inclusion benchmark results.

    Behaves like load_benches but expects inclusion runs (method 'incl').
    Returns a dataframe with columns:
      - 'benchmark', 'name'
      - for each tool: '<tool>-result' (true/false/TO/ERR/MISSING if present) and '<tool>-runtime' (float seconds)
      - optional '<tool>-check' if present in inputs
    Non-numeric runtimes (TO/ERR/MISSING) are set to the timeout value and coerced to float.
    If a tool has only timeouts (and thus no '-result' column in the CSV), the column is added and filled with 'TO'.
    """
    dfs = dict()
    for bench in benches:
        input_data = ""
        for tool in tools:
            assert tool != ""
            input_data += read_latest_result_file(bench, "incl", tool, timeout)
        # Convert pyco_proc CSV text to a DataFrame
        buf = io.StringIO()
        old_stdout = sys.stdout
        # temporarily set PARAMS_NUM=2 for inclusion tasks (two params: A and B)
        old_params = getattr(pyco_proc, 'PARAMS_NUM', None)
        pyco_proc.PARAMS_NUM = 2
        try:
            sys.stdout = buf
            pyco_proc.proc_res(io.StringIO(input_data), Namespace(csv=True, html=False, text=False, tick=False))
        finally:
            sys.stdout = old_stdout
            # restore original PARAMS_NUM
            if old_params is None:
                delattr(pyco_proc, 'PARAMS_NUM')
            else:
                pyco_proc.PARAMS_NUM = old_params
        csv_output = buf.getvalue()
        df = pd.read_csv(io.StringIO(csv_output), sep=";", dtype='unicode')

        # Ensure result column exists even when a tool only timed out (pyco_proc then omits outputs)
        for tool in tools:
            if f"{tool}-result" not in df.columns:
                df[f"{tool}-result"] = "TO"

        df["benchmark"] = bench
        dfs[bench] = df

    # Combine and select columns similarly to load_benches but for inclusion 'result'
    all_dfs = pd.concat(dfs, ignore_index=True)
    
    base_columns = ["benchmark", "name"]
    tool_columns = []
    for tool in tools:
        tool_columns.extend([f"{tool}-result", f"{tool}-runtime"])

    df_runtime_result = all_dfs[base_columns + tool_columns].copy()

    # Coerce runtimes to numeric where possible. Replace non-numeric with timeout value,
    # then mark any numeric runtime strictly greater than timeout as a timeout ('TO')
    for tool in tools:
        runtimes_num = pd.to_numeric(df_runtime_result[f"{tool}-runtime"], errors='coerce')
        mask_non_numeric = runtimes_num.isna()
        # For non-numeric runtimes (e.g., 'TO','ERR','MISSING'), set to the timeout so
        # downstream numeric ops won't fail. We'll later mark true timeouts explicitly.
        df_runtime_result.loc[mask_non_numeric, f"{tool}-runtime"] = float(timeout)
        # Ensure column is numeric for comparison
        df_runtime_result[f"{tool}-runtime"] = pd.to_numeric(df_runtime_result[f"{tool}-runtime"], errors='coerce').astype(float)
        # Now find rows where the recorded numeric runtime exceeded the configured timeout.
        mask_exceeded = df_runtime_result[f"{tool}-runtime"] > float(timeout)
        if mask_exceeded.any():
            # Mark runtime as timeout token and also mark result as 'TO' for consistency.
            df_runtime_result.loc[mask_exceeded, f"{tool}-runtime"] = 'TO'
            # If result column exists, set to 'TO' as well to indicate timeout.
            if f"{tool}-result" in df_runtime_result.columns:
                df_runtime_result.loc[mask_exceeded, f"{tool}-result"] = 'TO'

    return normalize_inclusion_timeouts(df_runtime_result, tools)

def left_join_on_name_benchmark(df_left: pd.DataFrame, df_right: pd.DataFrame, suffixes=("_left", "_right")) -> pd.DataFrame:
    """Left-join two DataFrames on columns 'name' and 'benchmark'.

    The function preserves all rows from ``df_left`` and matches rows from
    ``df_right`` where both 'name' and 'benchmark' are equal. If a row from
    ``df_left`` has no match in ``df_right``, the additional columns coming
    from ``df_right`` are set to NaN (null).

    Args:
        df_left (pd.DataFrame): Left dataframe. Must contain columns 'name' and 'benchmark'.
        df_right (pd.DataFrame): Right dataframe. Must contain columns 'name' and 'benchmark'.
        suffixes (tuple, optional): Suffixes to apply to overlapping column names
            (other than 'name' and 'benchmark') in the left and right side, respectively.
            Defaults to ("_left", "_right").

    Returns:
        pd.DataFrame: Result of the left join.

    Raises:
        KeyError: If required key columns are missing from either dataframe.
    """
    required = {"name", "benchmark"}
    missing_left = required - set(df_left.columns)
    missing_right = required - set(df_right.columns)

    if missing_left:
        raise KeyError(f"df_left is missing required columns: {sorted(missing_left)}")
    if missing_right:
        raise KeyError(f"df_right is missing required columns: {sorted(missing_right)}")

    # Perform the merge. Pandas will fill unmatched right-side columns with NaN.
    return pd.merge(df_left, df_right, on=["name", "benchmark"], how="left", suffixes=suffixes)

def _prepare_scatter_data(df, x_tool, y_tool, col, xname=None, yname=None, treat_errors_as_timeouts=False, timeout=None):
    """Prepare data for scatter plots by setting up column names and copying dataframe.

    Args:
        df: Input dataframe
        x_tool: Tool name for x-axis
        y_tool: Tool name for y-axis
        col: Column type ("runtime" or "states")
        xname: Custom x-axis name (optional)
        yname: Custom y-axis name (optional)
        treat_errors_as_timeouts: If True and col=='runtime', non-numeric tokens
            like 'TO','ERR','MISSING' will be treated as timeout and replaced
            with the numeric value provided in `timeout` before returning.
        timeout: numeric timeout value used when replacing error tokens.

    Returns:
        tuple: (prepared_df, x_col, y_col, xname, yname)
    """
    if xname is None:
        xname = x_tool
    if yname is None:
        yname = y_tool

    x_col = f"{x_tool}-{col}"
    y_col = f"{y_tool}-{col}"

    # work on a copy so we don't mutate the caller's dataframe
    df = df.copy(deep=True)

    # Handle numeric coercion. For runtimes we optionally replace non-numeric/error
    # tokens with the provided timeout so they appear as timeouts in plots.
    if col == 'runtime' and treat_errors_as_timeouts and timeout is not None:
        # Coerce to numeric, identify non-numeric entries and replace them with timeout
        x_num = pd.to_numeric(df[x_col], errors='coerce')
        y_num = pd.to_numeric(df[y_col], errors='coerce')
        mask_x_na = x_num.isna()
        mask_y_na = y_num.isna()
        x_num[mask_x_na] = float(timeout)
        y_num[mask_y_na] = float(timeout)
        df[x_col] = x_num.astype(float)
        df[y_col] = y_num.astype(float)
    else:
        # Default behaviour: coerce plotting columns to numeric floats (NaN allowed)
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce').astype(float)
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').astype(float)

    return df, x_col, y_col, xname, yname

def _apply_scatter_points(scatter, x_col, y_col, color_column, color_by_benchmark, show_legend, point_size=2.0):
    """Apply scatter points and rug plots to a ggplot object.
    
    Args:
        scatter: ggplot object
        x_col: x-axis column name
        y_col: y-axis column name
        color_column: column to use for coloring
        color_by_benchmark: whether to color by benchmark
        show_legend: whether to show legend
        point_size: size of points
        
    Returns:
        ggplot object with points added
    """
    if color_by_benchmark:
        scatter += p9.aes(x=x_col, y=y_col, color=color_column)
        scatter += p9.geom_point(size=point_size, na_rm=True, show_legend=show_legend, raster=True)
        # rug plots
        scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True)
    else:
        scatter += p9.aes(x=x_col, y=y_col, color=color_column)
        scatter += p9.geom_point(size=point_size, na_rm=True, show_legend=show_legend, raster=True, color="orange")
        # rug plots
        scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True, color="orange")
    
    return scatter

def _apply_scatter_theme(scatter, width, height, transparent, show_legend, legend_width, color_by_benchmark, color_labels=None):
    """Apply common theme elements to scatter plot.
    
    Args:
        scatter: ggplot object
        width: figure width
        height: figure height  
        transparent: whether background should be transparent
        show_legend: whether to show legend
        legend_width: additional width for legend
        color_by_benchmark: whether to color by benchmark
        color_labels: optional ordered list of labels to map colors to (legend order)
        
    Returns:
        ggplot object with theme applied
    """
    if show_legend:
        width += legend_width
        
    scatter += p9.theme_bw()
    scatter += p9.theme(panel_grid_major=p9.element_line(color='#666666', alpha=0.5))
    scatter += p9.theme(panel_grid_minor=p9.element_blank())
    scatter += p9.theme(figure_size=(width, height))
    scatter += p9.theme(axis_text=p9.element_text(size=24, color="black"))
    scatter += p9.theme(axis_title=p9.element_text(size=24, color="black"))
    scatter += p9.theme(legend_text=p9.element_text(size=12))
    scatter += p9.theme(legend_key_width=2)
    
    # Apply brighter, more distinct colors for benchmarks
    if color_by_benchmark:
        # Custom bright and distinct color palette
        bright_colors = ["#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A", "#FFD700", "#A6CEE3", "#FB9A99", "#B2DF8A", "#FDBF6F", "#CAB2D6", "#FFFF99"]
        if color_labels is not None:
            # build a mapping from label -> color preserving provided order
            labels = list(color_labels)
            color_map = {str(lbl): bright_colors[i % len(bright_colors)] for i, lbl in enumerate(labels)}
            scatter += p9.scale_color_manual(values=color_map, breaks=labels)
        else:
            scatter += p9.scale_color_manual(values=bright_colors)
    
    if transparent:
        scatter += p9.theme(
            plot_background=p9.element_blank(),
            panel_background = p9.element_rect(alpha=0.0),
            panel_border = p9.element_rect(colour = "black"),
            legend_background=p9.element_rect(alpha=0.0),
            legend_box_background=p9.element_rect(alpha=0.0),
        )

    if not show_legend:
        scatter += p9.theme(legend_position='none')
        
    return scatter

def _add_scatter_reference_lines(scatter, clamp_domain, dash_pattern=(0, (6, 2))):
    """Add reference lines (diagonal, vertical, horizontal) to scatter plot.
    
    Args:
        scatter: ggplot object
        clamp_domain: domain limits [min, max]
        dash_pattern: line dash pattern
        
    Returns:
        ggplot object with reference lines added
    """
    scatter += p9.geom_abline(intercept=0, slope=1, linetype=dash_pattern)  # diagonal
    scatter += p9.geom_vline(xintercept=clamp_domain[1], linetype=dash_pattern)  # vertical rule
    scatter += p9.geom_hline(yintercept=clamp_domain[1], linetype=dash_pattern)  # horizontal rule
    return scatter


def get_ax_formatter(log: bool):
    """Return an axis tick formatter.

    For log=True returns a function that formats values into LaTeX-style
    scientific notation (e.g. $10^{3}$ or $2.5\times10^{3}$). For log=False
    returns a mizani numeric formatter.
    """
    if log:
        def ax_formatter(values):
            def _fmt(v):
                try:
                    v = float(v)
                except Exception:
                    return ''
                if v <= 0 or np.isnan(v):
                    return str(v)
                exp = int(np.round(np.log10(v)))
                pow10 = 10 ** exp
                if abs(v - pow10) <= 1e-12 * max(1.0, v):
                    return f"$10^{{{exp}}}$"
                mant = v / pow10
                return f"${mant:g}\\times10^{{{exp}}}$"

            try:
                return [_fmt(val) for val in values]
            except TypeError:
                return _fmt(values)

        return ax_formatter
    else:
        return mizani.custom_format('{:n}')

def scatter_plot(df, x_tool, y_tool, timeout=120, clamp=True, clamp_domain=[0.01, 120], xname=None, yname=None, log=True, width=6, height=6, show_legend=True, legend_width=2, file_name_to_save=None, transparent=False, color_by_benchmark=True, color_column="benchmark", legend_name_map=None, tool_name_map=None):
    """Returns scatter plot for runtime comparison between two tools.

    Args:
        df (Dataframe): Dataframe containing the values to plot
        x_tool (str): name of the tool for x-axis
        y_tool (str): name of the tool for y-axis
        timeout (int, optional): timeout value. Defaults to 120.
        clamp (bool, optional): Whether values outside of clamp_domain are cut off. Defaults to True.
        clamp_domain (list, optional): The min/max values to plot. Defaults to [0.01, 120].
        xname (str, optional): Name of the x axis. Defaults to None, uses x_tool.
        yname (str, optional): Name of the y axis. Defaults to None, uses y_tool.
        log (bool, optional): Use logarithmic scale. Defaults to True.
        width (int, optional): Figure width in inches. Defaults to 6.
        height (int, optional): Figure height in inches. Defaults to 6. Will be set equal to width to make plot square.
        show_legend (bool, optional): Print legend. Defaults to True.
        legend_width (int, optional): Additional width for legend. Defaults to 2.
        file_name_to_save (str, optional): If not None, save the result to file_name_to_save.pdf. Defaults to None.
        transparent (bool, optional): Whether the generated plot should have transparent background. Defaults to False.
        color_by_benchmark (bool, optional): Whether the dots should be colored based on the benchmark. Defaults to True.
        color_column (str, optional): Name of the column to use for coloring. Defaults to 'benchmark'.
        legend_name_map (dict, optional): Optional dict mapping original benchmark names to labels shown in legend.
        tool_name_map (dict, optional): Optional dict mapping tool identifiers to display names. If provided
            and xname / yname are None, the axis labels will use tool_name_map[x_tool] / tool_name_map[y_tool]
            when available.
    """
    assert len(clamp_domain) == 2

    POINT_SIZE = 3.0
    DASH_PATTERN = (0, (6, 2))

    # Make the plot square by setting height equal to width
    height = width

    # Apply automatic axis name mapping if explicit names not given
    if xname is None and tool_name_map and x_tool in tool_name_map:
        xname = tool_name_map[x_tool]
    if yname is None and tool_name_map and y_tool in tool_name_map:
        yname = tool_name_map[y_tool]

    # Prepare data; treat non-numeric runtime tokens (ERR/MISSING/TO) as timeouts so
    # they are plotted at the timeout boundary.
    df, x_col, y_col, xname, yname = _prepare_scatter_data(df, x_tool, y_tool, "runtime", xname, yname, treat_errors_as_timeouts=True, timeout=timeout)

    # optionally create a legend mapping column
    color_col_used = color_column
    color_labels = None
    if color_by_benchmark and legend_name_map is not None and color_column in df.columns:
        legend_col = f"{color_column}"
        df[legend_col] = df[color_column].map(legend_name_map).fillna(df[color_column])
        color_col_used = legend_col
        # preserve order of appearance in dataframe for labels
        color_labels = list(pd.Series(df[color_col_used].unique()).astype(str))

    # formatter for axes' labels
    ax_formatter = mizani.custom_format('{:n}')

    if clamp:  # clamp overflowing values if required
        clamp_domain[1] = timeout
        df.loc[df[x_col] > clamp_domain[1], x_col] = clamp_domain[1]
        df.loc[df[y_col] > clamp_domain[1], y_col] = clamp_domain[1]

    # generate scatter plot
    scatter = p9.ggplot(df)
    scatter = _apply_scatter_points(scatter, x_col, y_col, color_col_used, color_by_benchmark, show_legend, POINT_SIZE)
    scatter += p9.labs(x=xname, y=yname)

    if log:  # log scale
        scatter += p9.scale_x_log10(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_log10(limits=clamp_domain, labels=ax_formatter)
    else:
        scatter += p9.scale_x_continuous(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_continuous(limits=clamp_domain, labels=ax_formatter)

    scatter = _apply_scatter_theme(scatter, width, height, transparent, show_legend, legend_width, color_by_benchmark, color_labels=color_labels)
    scatter = _add_scatter_reference_lines(scatter, clamp_domain, DASH_PATTERN)

    # When the legend is hidden, the plotting panel may become slightly rectangular due to
    # default margin calculations. Enforce a square panel (useful for diagonal comparison)
    # by setting an equal coordinate system.
    if not show_legend:
        try:
            scatter += p9.coord_equal()  # keeps aspect ratio 1:1 in data units
        except Exception:
            # Fallback: enforce panel aspect ratio via theme (in case coord_equal not available)
            scatter += p9.theme(aspect_ratio=1)

    if file_name_to_save != None:
        scatter.save(filename=f"{file_name_to_save}.pdf", dpi=500, verbose=False)

    return scatter

def scatter_plot_states(df, x_tool, y_tool, clamp=True, clamp_domain=None, xname=None, yname=None, log=True, width=6, height=6, show_legend=True, legend_width=2, file_name_to_save=None, transparent=False, color_by_benchmark=True, color_column="benchmark", legend_name_map=None, tool_name_map=None):
    """Returns scatter plot for state space size comparison between two tools.

    Args:
        df (Dataframe): Dataframe containing the values to plot
        x_tool (str): name of the tool for x-axis
        y_tool (str): name of the tool for y-axis
        clamp (bool, optional): Whether values outside of clamp_domain are cut off. Defaults to True.
        clamp_domain (list, optional): The min/max values to plot. Defaults to None (auto-computed from data).
        xname (str, optional): Name of the x axis. Defaults to None, uses x_tool.
        yname (str, optional): Name of the y axis. Defaults to None, uses y_tool.
        log (bool, optional): Use logarithmic scale. Defaults to True.
        width (int, optional): Figure width in inches. Defaults to 6.
        height (int, optional): Figure height in inches. Defaults to 6. Will be set equal to width to make plot square.
        show_legend (bool, optional): Print legend. Defaults to True.
        legend_width (int, optional): Additional width for legend. Defaults to 2.
        file_name_to_save (str, optional): If not None, save the result to file_name_to_save.pdf. Defaults to None.
        transparent (bool, optional): Whether the generated plot should have transparent background. Defaults to False.
        color_by_benchmark (bool, optional): Whether the dots should be colored based on the benchmark. Defaults to True.
        color_column (str, optional): Name of the column to use for coloring. Defaults to 'benchmark'.
        legend_name_map (dict, optional): Optional dict mapping original benchmark names to labels shown in legend.
        tool_name_map (dict, optional): Optional dict mapping tool identifiers to display names. If provided
            and xname / yname are None, the axis labels will use tool_name_map[x_tool] / tool_name_map[y_tool]
            when available.
    """
    POINT_SIZE = 3.0
    DASH_PATTERN = (0, (6, 2))

    # Make the plot square by setting height equal to width
    height = width

    # Apply automatic axis name mapping if explicit names not given
    if xname is None and tool_name_map and x_tool in tool_name_map:
        xname = tool_name_map[x_tool]
    if yname is None and tool_name_map and y_tool in tool_name_map:
        yname = tool_name_map[y_tool]

    # Prepare data
    df, x_col, y_col, xname, yname = _prepare_scatter_data(df, x_tool, y_tool, "states", xname, yname)

    # optionally create a legend mapping column
    color_col_used = color_column
    color_labels = None
    if color_by_benchmark and legend_name_map is not None and color_column in df.columns:
        legend_col = f"{color_column}"
        df[legend_col] = df[color_column].map(legend_name_map).fillna(df[color_column])
        color_col_used = legend_col
        color_labels = list(pd.Series(df[color_col_used].unique()).astype(str))

    # Auto-compute clamp domain if not provided
    if clamp_domain is None:
        # Only consider numeric values for computing domain
        x_numeric = df[x_col].dropna()
        y_numeric = df[y_col].dropna()
        if len(x_numeric) > 0 and len(y_numeric) > 0:
            max_states = max(x_numeric.max(), y_numeric.max())
            if log:
                clamp_domain = [1, max_states * 1.1]  # Start from 1 for log scale, add 10% margin
            else:
                clamp_domain = [0, max_states * 1.1]  # Start from 0 for linear scale
        else:
            clamp_domain = [1, 1000] if log else [0, 1000]  # fallback

    assert len(clamp_domain) == 2

    # formatter for axes' labels (shared helper)
    ax_formatter = get_ax_formatter(log)

    # Replace timeout/missing values with the maximum value (where dashed line is rendered)
    # Consider true missing (NaN) and empty strings as missing as well
    x_col_orig = f"{x_tool}-states"
    y_col_orig = f"{y_tool}-states"

    # Identify timeout/missing values. Handle NaN, empty string and tokens like 'TO','ERR','MISSING'
    x_series = df[x_col_orig]
    y_series = df[y_col_orig]
    x_str = x_series.astype(str).str.strip()
    y_str = y_series.astype(str).str.strip()
    x_timeouts = x_series.isna() # | x_str.isin(['TO', 'ERR', 'MISSING']) | (x_str == '')
    y_timeouts = y_series.isna() # | y_str.isin(['TO', 'ERR', 'MISSING']) | (y_str == '')
    
    # Replace timeout values with the maximum domain value
    df.loc[x_timeouts, x_col] = clamp_domain[1]
    df.loc[y_timeouts, y_col] = clamp_domain[1]

    if clamp:  # clamp overflowing values if required
        df.loc[df[x_col] > clamp_domain[1], x_col] = clamp_domain[1]
        df.loc[df[y_col] > clamp_domain[1], y_col] = clamp_domain[1]
        # For states, ensure minimum value is at least the lower bound
        if log:
            # For log scale, replace 0 and negative values with the minimum domain value
            df.loc[df[x_col] <= 0, x_col] = clamp_domain[0]
            df.loc[df[y_col] <= 0, y_col] = clamp_domain[0]
        else:
            # For linear scale, ensure values are not below 0
            df.loc[df[x_col] < 0, x_col] = 0
            df.loc[df[y_col] < 0, y_col] = 0

    # generate scatter plot
    scatter = p9.ggplot(df)
    scatter = _apply_scatter_points(scatter, x_col, y_col, color_col_used, color_by_benchmark, show_legend, POINT_SIZE)
    scatter += p9.labs(x=xname, y=yname)

    if log:  # log scale
        scatter += p9.scale_x_log10(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_log10(limits=clamp_domain, labels=ax_formatter)
    else:
        scatter += p9.scale_x_continuous(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_continuous(limits=clamp_domain, labels=ax_formatter)

    scatter = _apply_scatter_theme(scatter, width, height, transparent, show_legend, legend_width, color_by_benchmark, color_labels=color_labels)
    scatter = _add_scatter_reference_lines(scatter, clamp_domain, DASH_PATTERN)

    # Maintain a square panel when legend is suppressed for clearer state comparisons.
    if not show_legend:
        try:
            scatter += p9.coord_equal()
        except Exception:
            scatter += p9.theme(aspect_ratio=1)

    if file_name_to_save != None:
        scatter.save(filename=f"{file_name_to_save}.pdf", dpi=500, verbose=False)

    return scatter

def cactus_plot(df, tools, timeout = 120, tool_names = None, start = 0, end = None, logarithmic_y_axis=True, width=6, height=6, show_legend=True, put_legend_outside=False, file_name_to_save=None, num_of_x_ticks=5):
    """Returns cactus plot (sorted runtimes of each tool in tools). To print the result use result.figure.savefig("name_of_file.pdf", transparent=True).

    Args:
        df (Dataframe): Dataframe containing for each tool in tools column tool-result and tool-runtime containing the result and runtime for each benchmark.
        tools (list): List of tools to plot.
        tool_names (dict, optional): Maps each tool to its name that is used in the legend. If not set (=None), the names are taken directly from tools.
        start (int, optional): The starting position of the x-axis. Defaults to 0.
        end (int, optional): The ending position of the x-axis. If not set (=None), defaults to number of benchmarks, i.e. len(df).
        logarithmic_y_axis (bool, optional): Use logarithmic scale for the y-axis. Defaults to True.
        width (int, optional): Figure width in inches. Defaults to 6.
        height (int, optional): Figure height in inches. Defaults to 6.
        show_legend (bool, optional): Print legend. Defaults to True.
        put_legend_outside (bool, optional): Whether to put legend outside the plot. Defaults to False.
        file_name_to_save (str, optional): If not None, save the result to file_name_to_save.pdf. Defaults to None.
        num_of_x_ticks (int, optional): Number of ticks on the x-axis. Defaults to 5.
    """
    if tool_names == None:
        tool_names = { tool:tool for tool in tools }

    if end == None:
        end = len(df)

    concat = dict()

    for tool in tools:
        name = tool_names[tool]

        concat[name] = pd.Series(sorted(get_solved(df, tool)[tool + "-runtime"].tolist()))

    concat = pd.DataFrame(concat)


    plt = concat.plot.line(figsize=(width, height))
    ticks = np.linspace(start, end, num_of_x_ticks, dtype=int)
    plt.set_xticks(ticks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.set_xlim([start, end])
    plt.set_ylim([0.1, timeout])
    if logarithmic_y_axis:
        plt.set_yscale('log')
    plt.set_xlabel("Instances", fontsize=16)
    plt.set_ylabel("Runtime [s]", fontsize=16)

    if show_legend:
        if put_legend_outside:
            plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left',framealpha=0.1)
        else:
            plt.legend(loc='upper left',framealpha=0.1)

        # plt.axvline(x=end)

        # plt.get_legend().remove()
        # figlegend = pylab.figure(figsize=(4,4))
        # figlegend.legend(plt.get_children(), concat.columns, loc='center', frameon=False)
        # figlegend.savefig(f"graphs/fig-cactus-{file_name}-legend.pdf", dpi=1000, bbox_inches='tight')
        # plt.figure.savefig(f"graphs/fig-cactus-{file_name}.pdf", dpi=1000, bbox_inches='tight')

    plt.figure.tight_layout()
    if file_name_to_save != None:
        plt.figure.savefig(f"{file_name_to_save}.pdf", transparent=True)
    return plt

def sanity_check(df, tool, compare_with):
    """Returns dataframe containing rows of df, where df[tool-result] is different (sat vs. unsat) than the result of any of the tools in compare_with

    Args:
        compare_with (list): List of tools to compare with.
    """
    all_bad = []
    for tool_other in compare_with:
        pt = df
        pt = pt[((pt[tool+"-result"].str.strip() == 'true') & (pt[tool_other+"-result"].str.strip() == 'false') | (pt[tool+"-result"].str.strip() == 'false') & (pt[tool_other+"-result"].str.strip() == 'true'))]
        all_bad.append(pt)
    return pd.concat(all_bad).drop_duplicates()

def get_solved_incl(df, tool):
    """Return rows where the inclusion tool produced a definitive result (true or false).

    Args:
        df (pd.DataFrame): Dataframe containing inclusion results with a column
            named "<tool>-result" for the given tool.
        tool (str): Tool name prefix used in the dataframe columns.

    Returns:
        pd.DataFrame: Subset of `df` where `<tool>-result` equals 'true' or 'false'
        (string comparison, whitespace-insensitive). The original dataframe's
        row order and other columns are preserved.
    """
    return df[(df[tool+"-result"].str.strip() == 'true')|(df[tool+"-result"].str.strip() == 'false')]

def normalize_inclusion_timeouts(df, tools):
    """Normalize inclusion timeouts by setting runtime to 'TO' for unsolved instances.
    
    For each tool in the provided list, this function:
    1. Sets '<tool>-runtime' to 'TO' for rows where:
       - The tool did not produce a definitive result (i.e., rows not in get_solved_incl(df, tool))
       - AND the current '<tool>-runtime' value is numeric
    2. Sets '<tool>-result' to 'ERR' for rows where '<tool>-result' is NaN
    
    This preserves existing non-numeric values like 'ERR' or 'MISSING' while only
    converting numeric runtimes for unsolved instances to 'TO'.
    
    Args:
        df (pd.DataFrame): Dataframe containing inclusion results with columns
            '<tool>-result' and '<tool>-runtime' for each tool.
        tools (list): List of tool name prefixes to normalize.
    
    Returns:
        pd.DataFrame: Modified dataframe with normalized runtime values. 
        The function modifies the dataframe in place and also returns it.
    
    Example:
        >>> df = normalize_inclusion_timeouts(df, ['kofola-c729572', 'spot-2.14.2'])
    """
    for tool in tools:
        result_col = f"{tool}-result"
        runtime_col = f"{tool}-runtime"
        
        # Skip if columns don't exist
        if result_col not in df.columns or runtime_col not in df.columns:
            continue
        
        # Set result to 'ERR' where it is NaN
        result_nan_mask = df[result_col].isna()
        df.loc[result_nan_mask, result_col] = 'ERR'
        
        # Get indices of solved instances
        solved_indices = get_solved_incl(df, tool).index
        
        # Identify unsolved rows
        unsolved_mask = ~df.index.isin(solved_indices)
        
        # Check which runtime values are numeric
        runtime_numeric = pd.to_numeric(df[runtime_col], errors='coerce')
        is_numeric_mask = runtime_numeric.notna()
        
        # Set runtime to 'TO' only for rows that are unsolved AND have numeric runtime
        mask_to_normalize = unsolved_mask & is_numeric_mask
        df.loc[mask_to_normalize, runtime_col] = 'TO'
    
    return df

def get_timeouts_incl(df, tool):
    """Return rows where the inclusion tool timed out.

    Args:
        df (pd.DataFrame): Dataframe containing inclusion results with a column
            named "<tool>-result" for the given tool.
        tool (str): Tool name prefix used in the dataframe columns.

    Returns:
        pd.DataFrame: Subset of `df` where `<tool>-result` equals 'TO'
        (timeout token). Comparison is done on the stripped string value.
    """
    return df[(df[tool+"-result"].str.strip() == 'TO')]

def get_errors_incl(df, tool):
    """Return rows where the inclusion tool produced an error or missing result.

    Args:
        df (pd.DataFrame): Dataframe containing inclusion results with a column
            named "<tool>-result" for the given tool.
        tool (str): Tool name prefix used in the dataframe columns.

    Returns:
        pd.DataFrame: Subset of `df` where `<tool>-result` equals 'ERR' or
        'MISSING' (string comparison, whitespace-insensitive).
    """
    return df[(df[tool+"-result"].str.strip() == 'ERR')|(df[tool+"-result"].str.strip() == 'MISSING')]

def get_missing_incl(df, tool):
    """Return rows where both the tool's result and runtime are truly missing (null).

    A row is considered "missing" if:
      - The column `<tool>-result` is NaN/null, and
      - The runtime column is NaN/null. We look for `<tool>-time` first (if present),
        otherwise fall back to `<tool>-runtime`.

    Args:
        df (pd.DataFrame): DataFrame with inclusion results.
        tool (str): Tool prefix used in the column names.

    Returns:
        pd.DataFrame: Subset of `df` where both result and runtime are null.
                       If required columns are not present, returns an empty slice.
    """
    result_col = f"{tool}-result"
    # Prefer "-time" if it exists, else use "-runtime"
    time_col = f"{tool}-time" if f"{tool}-time" in df.columns else (f"{tool}-runtime" if f"{tool}-runtime" in df.columns else None)

    if result_col not in df.columns or time_col is None:
        # Return empty dataframe with same columns if we cannot evaluate the condition
        return df.iloc[0:0]

    return df[df[result_col].isna() & df[time_col].isna()]

def get_solved(df, tool):
    """Returns dataframe containing rows of df where <tool>-states is numeric (i.e., solved).

    Non-numeric values like 'TO' or 'ERR' are excluded. The returned dataframe has the
    <tool>-states column converted to integers.
    """
    states_col = pd.to_numeric(df[f"{tool}-states"], errors='coerce')
    mask = states_col.notna()
    df_solved = df[mask].copy()
    # convert the states column to integer type for the returned dataframe
    df_solved[f"{tool}-states"] = states_col[mask].astype(int)
    return df_solved

def get_timeouts(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is timeout, i.e., 'TO'"""
    return df[(df[tool+"-states"].str.strip() == 'TO')]

def get_invalid(df, check_tools):
    """Returns dataframe containing rows of df where any of the check_tools has check column set to False.
    
    Args:
        df (Dataframe): data
        check_tools (list): List of tools to check for invalid results.
        
    Returns:
        Dataframe: Rows where at least one tool in check_tools has check=False
    """
    if not check_tools:
        return df.iloc[0:0]  # Return empty dataframe with same columns
    
    # Build condition for any tool having False check, but exclude rows where the corresponding
    # <tool>-states value is missing (NaN, empty string or literal 'MISSING').
    conditions = []
    for tool in check_tools:
        check_col = f"{tool}-check"
        check_state = f"{tool}-states"
        if check_col in df.columns and check_state in df.columns:
            # original states series (may contain NaN or special tokens like 'MISSING')
            state_series = df[check_state]
            # normalized string version for comparisons
            state_str = state_series.astype(str).str.strip()
            # consider a state valid if it's not NaN, not empty and not the token 'MISSING'
            state_valid = state_series.notna() & (state_str.str.upper() != 'MISSING') & (state_str != '')

            # Check for False values in the check column (string 'False' or boolean False)
            check_false = (df[check_col].astype(str).str.strip().str.lower() == 'false') | (df[check_col] == False)

            # only mark invalid when check says False and the state is present/valid
            condition = check_false & state_valid
            conditions.append(condition)
        elif check_col in df.columns:
            # If states column is absent, fall back to using the check column alone
            check_false = (df[check_col].astype(str).str.strip().str.lower() == 'false') | (df[check_col] == False)
            conditions.append(check_false)
    
    if not conditions:
        return df.iloc[0:0]  # Return empty dataframe if no check columns found
    
    # Combine all conditions with OR (any tool having False check)
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = combined_condition | condition
    
    return df[combined_condition]

def get_errors(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is error, i.e., 'ERR'"""
    return df[(df[tool+"-states"].str.strip().isin(['ERR', 'MISSING']))]

def simple_table(df, tools, benches, separately=False, stat_from_solved=True):
    """Prints a simple table with statistics for each tools.

    Args:
        df (Dataframe): data
        tools (list): List of tools.
        benches (list): List of benchmark sets.
        separately (bool, optional): Should we print table for each benchmark separately. Defaults to False.
        times_from_solved (bool, optional): Should we compute total, average and median time from only solved instances. Defaults to True.
    """

    result = ""

    def print_table_from_full_df(df):
        header = ["tool", "✅", "❌", "states", "max-states", "states-avg", "states-med", "time", "time-avg", "time-med", "TO", "ERR"]
        result = ""
        result += f"# of automata: {len(df)}\n"
        result += "----------------------------------------------------------------------------------------------------\n"
        table = [header]
        for tool in tools:
            valid = len(get_solved(df, tool))
            to = len(get_timeouts(df, tool))
            err = len(get_errors(df, tool))
            runtime_col = df[f"{tool}-runtime"]
            states_col = df[f"{tool}-states"]
            if stat_from_solved:
                runtime_col = get_solved(df, tool)[f"{tool}-runtime"]
                states_col = get_solved(df, tool)[f"{tool}-states"]
            states_total= states_col.sum()
            states_avg = states_col.mean()
            states_med = states_col.median()
            states_max = states_col.max()
            runtime_total = runtime_col.sum()
            runtime_avg = runtime_col.mean()
            runtime_med = runtime_col.median()
            
            table.append([tool, valid, to + err, states_total, states_max, states_avg, states_med, runtime_total, runtime_avg, runtime_med, to, err])
        result += tab.tabulate(table, headers='firstrow', floatfmt=".2f") + "\n"
        result += "----------------------------------------------------------------------------------------------------\n\n"
        return result

    if (separately):
        for bench in benches:
            result += f"Benchmark {bench}\n"
            result += print_table_from_full_df(df[df["benchmark"] == bench])
    else:
        result += print_table_from_full_df(df[df["benchmark"].isin(benches)])

    return result

def simple_table_incl(df, tools, benches, separately=False, stat_from_solved=True):
    """Generate a simple statistics table for inclusion benchmarks.

    The function computes per-tool counts and timing statistics for inclusion
    experiments. It expects `df` to contain columns named `<tool>-result`
    (values like 'true','false','TO','ERR','MISSING') and `<tool>-runtime`
    (seconds as numeric or non-numeric tokens).

    Args:
        df (pd.DataFrame): Combined dataframe with a 'benchmark' column and
            per-tool '<tool>-result' and '<tool>-runtime' columns.
        tools (list): List of tool name prefixes to include in the table.
        benches (list): List of benchmark names to consider (rows filtered by
            `df["benchmark"].isin(benches)` unless `separately` is True).
        separately (bool): If True, return a concatenated string containing a
            table for each benchmark in `benches`; otherwise produce a single
            table over all specified benchmarks.
        stat_from_solved (bool): When True, compute aggregate timing stats
            (total/mean/median) from only solved instances (i.e. rows where
            `<tool>-result` is 'true' or 'false'). When False, use all rows.

    Returns:
        str: A formatted ASCII table (tabulate) summarizing for each tool:
             [tool, #solved, #not-solved, total-time, avg-time, med-time, TO, ERR]
    """

    result = ""

    def print_table_from_full_df(df):
        header = ["tool", "✅", "❌", "time", "time-avg", "time-med", "TO", "ERR", "MISSING"]
        result = ""
        result += f"# of automata: {len(df)}\n"
        result += "----------------------------------------------------------------------------------------------------\n"
        table = [header]
        for tool in tools:
            valid = len(get_solved_incl(df, tool))
            to = len(get_timeouts_incl(df, tool))
            err = len(get_errors_incl(df, tool))
            miss = len(get_missing_incl(df, tool))
            runtime_col = df[f"{tool}-runtime"]
            if stat_from_solved:
                runtime_col = get_solved_incl(df, tool)[f"{tool}-runtime"]
            runtime_total = runtime_col.sum()
            runtime_avg = runtime_col.mean()
            runtime_med = runtime_col.median()

            table.append([tool, valid, to + err, runtime_total, runtime_avg, runtime_med, to, err, miss])
        result += tab.tabulate(table, headers='firstrow', floatfmt=".2f") + "\n"
        result += "----------------------------------------------------------------------------------------------------\n\n"
        return result

    if (separately):
        for bench in benches:
            result += f"Benchmark {bench}\n"
            result += print_table_from_full_df(df[df["benchmark"] == bench])
    else:
        result += print_table_from_full_df(df[df["benchmark"].isin(benches)])

    return result


def build_inclusion_stats_df(df, tools, benches, stat_from_solved=True, numeric_timeout_fallback=True, tool_for_comparison=None):
    """Return a DataFrame with the same statistics as printed by simple_table_incl.

    This simplified version always aggregates across all provided benches (no per-benchmark
    breakdown). The resulting DataFrame columns replicate the header used in
    simple_table_incl:
        ["tool", "✅", "❌", "time", "time-avg", "time-med", "TO", "ERR", "MISSING"].

    Column semantics:
        ✅ (solved): count of rows where <tool>-result in {'true','false'}.
        ❌ (not-solved): TO + ERR counts (MISSING reported separately).
        time / time-avg / time-med: aggregate runtime statistics either over solved rows
            (stat_from_solved=True) or all rows. Non-numeric tokens ('TO','ERR','MISSING', etc.)
            are coerced to NaN and ignored. If all values are NaN, total is 0.0 and avg/median NaN.
        TO / ERR / MISSING: counts derived via helper functions.
        wins / loses: If tool_for_comparison is provided, counts instances where this tool's
            runtime is strictly better (wins) or worse (loses) than tool_for_comparison.
            Only instances where both tools have numeric runtimes are considered.

    Args:
        df (pd.DataFrame): Inclusion results containing at least 'benchmark' and tool columns.
        tools (list[str]): Tool prefixes to summarize.
        benches (list[str]): Benchmarks to include (rows filtered by df['benchmark'].isin(benches)).
        stat_from_solved (bool): If True, restrict runtime aggregates to solved instances.
        numeric_timeout_fallback (bool): Retained for backwards compatibility; when False no
            special handling (non-numerics already dropped). Currently both code paths behave identically.
        tool_for_comparison (str, optional): Tool to compare against for wins/loses statistics.
            If None, wins and loses columns are omitted.

    Returns:
        pd.DataFrame: One row per tool with the described statistics.
    """

    # Guard: ensure required columns exist for each tool; if missing, create empty ones to avoid KeyErrors
    needed_cols = []
    for tool in tools:
        needed_cols.extend([f"{tool}-result", f"{tool}-runtime"])
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        # Create the columns with NaN so helper functions just yield zero counts
        for col in missing:
            df[col] = pd.NA

    def _one_scope(scope_df, scope_name=None):
        rows = []
        for tool in tools:
            solved_cnt = len(get_solved_incl(scope_df, tool))
            to_cnt = len(get_timeouts_incl(scope_df, tool))
            err_cnt = len(get_errors_incl(scope_df, tool))
            miss_cnt = len(get_missing_incl(scope_df, tool))

            # Runtime selection
            if stat_from_solved:
                runtime_series = get_solved_incl(scope_df, tool)[f"{tool}-runtime"]
            else:
                runtime_series = scope_df[f"{tool}-runtime"]

            # Coerce to numeric; drop non-numeric (timeouts/errors) naturally via NaN
            runtime_numeric = pd.to_numeric(runtime_series, errors='coerce')
            # If desired, optionally replace NaN that came from non-numeric tokens with the timeout value –
            # but current semantics (like simple_table_incl) effectively ignored string tokens in sums.
            if not numeric_timeout_fallback:
                # Keep numeric only; nothing else to do (already NaN for non-numeric)
                pass

            time_total = runtime_numeric.sum(min_count=1)  # min_count ensures empty sum -> NaN
            time_avg = runtime_numeric.mean()
            time_med = runtime_numeric.median()

            # For consistency with old behaviour, if there were simply no numeric entries (all NaN),
            # treat total as 0 (optional design choice).
            if pd.isna(time_total):
                time_total = 0.0

            row = {
                'tool': tool,
                'solved': solved_cnt,
                'time': time_total,
                'avg': time_avg,
                'med': time_med,
                'OOR': to_cnt + err_cnt,
                'missing': miss_cnt,
            }
            
            # Compute wins and loses if tool_for_comparison is provided
            if tool_for_comparison is not None:
                # Get indices of missing rows for both tools (to exclude from comparison)
                tool_missing_indices = get_missing_incl(scope_df, tool).index
                comp_missing_indices = get_missing_incl(scope_df, tool_for_comparison).index
                
                # Create a mask to exclude rows where either tool has missing data
                not_missing_mask = ~scope_df.index.isin(tool_missing_indices) & ~scope_df.index.isin(comp_missing_indices)
                
                # Filter the dataframe to exclude missing rows
                comparison_df = scope_df[not_missing_mask]
                
                # Get numeric runtimes for both tools (only for non-missing rows)
                tool_runtime = pd.to_numeric(comparison_df[f"{tool}-runtime"], errors='coerce')
                comp_runtime = pd.to_numeric(comparison_df[f"{tool_for_comparison}-runtime"], errors='coerce')
                
                # Identify valid numeric runtimes for each tool
                tool_has_runtime = tool_runtime.notna()
                comp_has_runtime = comp_runtime.notna()
                
                # Wins: 
                # 1. Both have numeric runtime and this tool is strictly faster
                # 2. This tool has numeric runtime but comparison tool doesn't (ERR/TO)
                wins_both_valid = (tool_has_runtime & comp_has_runtime & (tool_runtime < comp_runtime)).sum()
                wins_comp_failed = (tool_has_runtime & ~comp_has_runtime).sum()
                wins = wins_both_valid + wins_comp_failed
                
                # Loses: 
                # 1. Both have numeric runtime and this tool is strictly slower
                # 2. Comparison tool has numeric runtime but this tool doesn't (ERR/TO)
                loses_both_valid = (tool_has_runtime & comp_has_runtime & (tool_runtime > comp_runtime)).sum()
                loses_tool_failed = (~tool_has_runtime & comp_has_runtime).sum()
                loses = loses_both_valid + loses_tool_failed
                
                row['wins'] = loses
                row['loses'] = wins
            
            if scope_name is not None:
                row['benchmark'] = scope_name
            rows.append(row)
        columns = ['tool', 'solved', 'time', 'avg', 'med', 'OOR']
        if tool_for_comparison is not None:
            columns.extend(['wins', 'loses'])
        columns.extend(['missing'])
        if scope_name is not None:
            columns = ['benchmark'] + columns
        return pd.DataFrame(rows, columns=columns)

    benches = list(benches)
    scope_df = df[df['benchmark'].isin(benches)]
    return _one_scope(scope_df)


def build_complement_stats_df(df, tools, benches, stat_from_solved=True):
    """Build a statistics DataFrame for complementation benchmarks (no time columns).

    This mirrors the logic of simple_table (non-inclusion) but omits every time-based
    column. It summarizes state-related measures per tool across the selected benches.

    Output columns (one row per tool):
        tool          : tool name
        solved        : number of solved instances (numeric <tool>-states)
        total         : sum of states over considered instances
        max           : maximum number of states
        avg           : average number of states
        med           : median number of states
        OOR           : count of out-of-result instances (timeouts + errors)

    Args:
        df (pd.DataFrame): Complementation results containing per-tool '<tool>-states' columns.
        tools (list[str]): Tool prefixes.
        benches (list[str]): Benchmarks to include.
        stat_from_solved (bool): If True (default) compute state aggregates (sum/avg/med/max)
            over solved instances only; otherwise attempt to include all numeric entries
            (non-numeric tokens are ignored via NaN coercion).

    Returns:
        pd.DataFrame: Summary statistics per tool.
    """

    # Filter relevant benches
    scope_df = df[df['benchmark'].isin(benches)]

    rows = []
    for tool in tools:
        # Identify solved / timeout / error counts
        solved_df = get_solved(scope_df, tool)
        solved_cnt = len(solved_df)
        to_cnt = len(get_timeouts(scope_df, tool))
        err_cnt = len(get_errors(scope_df, tool))
        oor_cnt = to_cnt + err_cnt

        # Choose data for state statistics
        if stat_from_solved:
            state_series = solved_df[f"{tool}-states"]
        else:
            # Use any numeric states (coerce errors/timeouts to NaN and drop)
            state_series = pd.to_numeric(scope_df[f"{tool}-states"], errors='coerce')
            state_series = state_series.dropna()

        # If there are no numeric entries, produce zeros/NaN appropriately
        if len(state_series) == 0:
            states_total = 0
            states_max = 0
            states_avg = float('nan')
            states_med = float('nan')
        else:
            # Ensure numeric dtype
            state_series = pd.to_numeric(state_series, errors='coerce')
            states_total = state_series.sum()
            states_max = state_series.max()
            states_avg = state_series.mean()
            states_med = state_series.median()

        rows.append({
            'tool': tool,
            'solved': solved_cnt,
            'total': states_total,
            'max': states_max,
            'avg': states_avg,
            'med': states_med,
            'OOR': oor_cnt,
        })

    columns = ['tool', 'solved', 'total', 'max', 'avg', 'med', 'OOR']
    return pd.DataFrame(rows, columns=columns)


def complement_stats_to_latex(stats_df, index_column='tool', float_format="{:.2f}", caption=None, label=None,
                              column_format=None, escape=True):
    """Convert a complementation statistics DataFrame to LaTeX (no bolding logic).

    The input should be produced by build_complement_stats_df. All numeric columns are
    formatted uniformly; no emphasis (bold) is applied.

    Args:
        stats_df (pd.DataFrame): DataFrame with columns like 'states-total','states-max', etc.
        index_column (str): Column to set as index prior to LaTeX export (default 'tool').
        float_format (str): Format for floating point columns (avg/med/max). Total is rendered
            as integer if it is integral; otherwise same float format.
        caption (str|None): Optional LaTeX caption.
        label (str|None): Optional LaTeX label (added as tab:<label> if provided).
        column_format (str|None): Optional LaTeX column format passed through to pandas.
        escape (bool): Whether to escape LaTeX special characters.

    Returns:
        str: LaTeX tabular string.
    """
    df_latex = stats_df.copy()

    # Identify numeric columns we expect
    numeric_cols = [c for c in ['total', 'max', 'avg', 'med'] if c in df_latex.columns]
    for col in numeric_cols:
        df_latex[col] = pd.to_numeric(df_latex[col], errors='coerce')

    # Format totals as integers when possible; otherwise use float_format
    if 'total' in df_latex.columns:
        df_latex['total'] = df_latex['total'].map(
            lambda v: ('' if pd.isna(v) else (str(int(v)) if float(v).is_integer() else float_format.format(v)))
        )
    # Format remaining float columns uniformly
    for col in ['max', 'avg', 'med']:
        if col in df_latex.columns:
            df_latex[col] = df_latex[col].map(lambda v: (float_format.format(v) if pd.notna(v) else ''))

    if index_column in df_latex.columns:
        df_latex = df_latex.set_index(index_column)

    return df_latex.to_latex(escape=escape, caption=caption, label=(f"tab:{label}" if label else None), column_format=column_format)


def inclusion_stats_to_latex(stats_df, index_column='tool', float_format="{:.2f}", caption=None, label=None,
                              column_format=None, escape=True, bold_best_time=False):
    """Convert an inclusion statistics DataFrame to a LaTeX tabular string.

    Expects a DataFrame produced by build_inclusion_stats_df. The function can optionally
    set an index column (default 'tool'), format floating point numbers, and (optionally)
    bold the best (minimum) value in the 'time' column per *benchmark* segment if that
    column exists.

    Args:
        stats_df (pd.DataFrame): Summary statistics (possibly including 'benchmark').
        index_column (str): Column to move to index before conversion (ignored if missing).
        float_format (str): Format spec used for float values.
        caption (str|None): Optional LaTeX caption.
        label (str|None): Optional LaTeX label (without leading 'tab:').
        column_format (str|None): Optional column format passed to pandas to_latex.
        escape (bool): Whether to escape LaTeX special chars.
        bold_best_time (bool): If True, bold the minimum 'time' per benchmark (or globally
            if no benchmark column) before export.

    Returns:
        str: Complete LaTeX table code (including tabular environment).
    """
    df_latex = stats_df.copy()

    # Optionally bold best times
    if bold_best_time and 'time' in df_latex.columns:
        def _bold_min(group):
            numeric = pd.to_numeric(group['time'], errors='coerce')
            if numeric.notna().any():
                min_val = numeric.min()
                # apply bold formatting using LaTeX \textbf{}
                group.loc[numeric == min_val, 'time'] = group.loc[numeric == min_val, 'time'].map(lambda v: f"\\textbf{{{float_format.format(v)}}}")
                # Format non-min numeric 'time' entries as plain numbers
                group.loc[numeric != min_val, 'time'] = group.loc[numeric != min_val, 'time'].map(lambda v: float_format.format(v))
            return group

        if 'benchmark' in df_latex.columns:
            df_latex = df_latex.groupby('benchmark', group_keys=False).apply(_bold_min)
        else:
            df_latex = _bold_min(df_latex)

    # Format other float columns (excluding 'time' already formatted if bolded)
    float_cols = [c for c in ['time', 'time-avg', 'time-med'] if c in df_latex.columns]
    for col in float_cols:
        # Skip if already string (e.g., bold formatted)
        if df_latex[col].dtype != object or not df_latex[col].astype(str).str.startswith('\\textbf').any():
            df_latex[col] = pd.to_numeric(df_latex[col], errors='coerce').map(lambda v: (float_format.format(v) if pd.notna(v) else ''))

    # Move index column if present
    if index_column in df_latex.columns:
        df_latex = df_latex.set_index(index_column)

    latex = df_latex.to_latex(escape=escape, caption=caption, label=(f"tab:{label}" if label else None), column_format=column_format)
    return latex


def add_vbs(df, tools_list, name = None):
    """Adds virtual best solvers from tools in tool_list

    Args:
        df (Dataframe): data
        tools_list (list): list of tools
        name (str, optional): Name of the vbs used for the new columns. If not set (=None), the name is generated from the name of tools in tool_list.

    Returns:
        Dataframe: same as df but with new columns for the vbs
    """
    if name == None:
        name = "+".join(tools_list)
    df[f"{name}-runtime"] = df[[f"{tool}-runtime" for tool in tools_list]].min(axis=1)
    def get_result(row):
        nonlocal tools_list
        if "sat" in [str(row[f"{tool}-result"]).strip() for tool in tools_list]:
            return "sat"
        elif "unsat" in [str(row[f"{tool}-result"]).strip() for tool in tools_list]:
            return "unsat"
        else:
            return "unknown"
    df[f"{name}-result"] = df.apply(get_result, axis=1) # https://stackoverflow.com/questions/26886653/create-new-column-based-on-values-from-other-columns-apply-a-function-of-multi
    return df

def write_latex_table_body(df, float_format="{:.2f}", format_index_name=True, index_to_latex=None):
    def format_index_name_default(name):
        if index_to_latex and name in index_to_latex:
            return index_to_latex[name]

        return name

    df_table = df
    if format_index_name:
        df_table = df.rename(index=format_index_name_default)
    return df_table.to_latex(buf=None, columns=None, header=False, index=True, na_rep='NaN', formatters=None, float_format=float_format.format, sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None, label=None, position=None).splitlines()

def parse_classification(benchmark_name):
    """Parse classification CSV file for a specific benchmark and return pandas DataFrame.
    
    Args:
        benchmark_name (str): Name of the benchmark (will look for classifications/<benchmark>-classification.csv)
        
    Returns:
        pd.DataFrame: DataFrame with columns 'automaton', 'benchmark', and 'info' where info contains
                     a dictionary of classification properties
    """
    classification_csv_path = f"classifications/{benchmark_name}-classification.csv"
    
    # Read the classification CSV
    df_class = pd.read_csv(classification_csv_path, sep=';')
    
    # Get property columns (all columns except 'name')
    property_columns = [col for col in df_class.columns if col != 'name']
    
    # Create info column by combining all property columns into a dictionary
    def create_info_dict(row):
        info = {}
        for prop in property_columns:
            # Convert to boolean (1 -> True, 0 -> False)
            info[prop] = bool(int(row[prop]))
        return info
    
    # Create the result DataFrame
    result_df = pd.DataFrame({
        'automaton': df_class['name'],
        'benchmark': benchmark_name,
        'info': df_class.apply(create_info_dict, axis=1)
    })
    
    return result_df

def join_with_classification(main_df, classification_df, automaton_column='name', benchmark_column='benchmark'):
    """Join main dataframe with classification dataframe on automaton name and benchmark.
    
    Args:
        main_df (pd.DataFrame): Main dataframe containing benchmark results
        classification_df (pd.DataFrame): Classification dataframe from parse_classification()
        automaton_column (str, optional): Column name in main_df containing automaton names. 
                                         Defaults to 'name'.
        benchmark_column (str, optional): Column name in main_df containing benchmark names.
                                         Defaults to 'benchmark'.
        
    Returns:
        pd.DataFrame: Merged dataframe with classification info added
    """
    # Perform left join to preserve all rows from main_df
    result_df = main_df.merge(
        classification_df, 
        left_on=[automaton_column, benchmark_column], 
        right_on=['automaton', 'benchmark'], 
        how='left'
    )
    
    # Drop the redundant columns from classification_df
    if 'automaton' in result_df.columns:
        result_df = result_df.drop('automaton', axis=1)
    
    return result_df

def parse_classifications_for_benchmarks(benchmark_names):
    """Parse classification CSV files for multiple benchmarks and return combined DataFrame.
    
    Args:
        benchmark_names (list): List of benchmark names
        
    Returns:
        pd.DataFrame: Combined DataFrame with columns 'automaton', 'benchmark', and 'info'
    """
    classification_dfs = []
    
    for benchmark_name in benchmark_names:
        try:
            df = parse_classification(benchmark_name)
            classification_dfs.append(df)
        except FileNotFoundError:
            print(f"WARNING: Classification file not found for benchmark '{benchmark_name}'")
            continue
        except Exception as e:
            print(f"WARNING: Error parsing classification for benchmark '{benchmark_name}': {e}")
            continue
    
    if not classification_dfs:
        print("WARNING: No classification files could be loaded")
        return pd.DataFrame(columns=['automaton', 'benchmark', 'info'])
    
    # Combine all classification dataframes
    combined_df = pd.concat(classification_dfs, ignore_index=True)
    
    return combined_df

def ba_bench_to_hoa(df_ba):
    """Convert BA benchmark instance names to their HOA counterparts.

    The input dataframe `df_ba` contains a column `name` whose value is a string
    of one or more file paths separated by '+', e.g. "<file1>+<file2>". This
    function splits the name, converts each part using `conv_ba_instance(part, bench)`,
    and joins the converted parts back with '+'.

    Notes:
      - The `df_hoa` parameter is accepted for compatibility but is not used
        by this conversion routine.

    Args:
        df_ba (pd.DataFrame): DataFrame with columns 'name' and optionally 'benchmark'.
        df_hoa (pd.DataFrame): Unused; kept for API compatibility.

    Returns:
        pd.DataFrame: Copy of `df_ba` with converted 'name' values and original BA names stored in 'ba_name' column.
    """
    if 'name' not in df_ba.columns:
        raise KeyError("df_ba must contain a 'name' column")

    # Work on a copy to avoid mutating the caller's dataframe
    df_out = df_ba.copy(deep=True)

    # Store original BA instance names in a new column
    df_out['ba_name'] = df_out['name']

    def _convert_row(row):
        bench = row['benchmark'] if 'benchmark' in row and pd.notna(row['benchmark']) else None
        raw = '' if pd.isna(row['name']) else str(row['name'])
        parts = [p.strip() for p in raw.split('+') if p.strip() != '']
        converted_parts = [conv_ba_instance(p, bench) for p in parts]
        return '+'.join(converted_parts)

    # Replace 'name' with converted HOA name(s), preserving row order
    df_out['name'] = df_out.apply(_convert_row, axis=1)

    return df_out

def conv_ba_instance(ba_file, bench):
    """Convert or map a BA filename to the corresponding HOA filename for a
    given benchmark collection.

    Current minimal behaviour:
      - For bench == 'rabit' simply change the filename extension from '.ba'
        to '.hoa' and return the new filename.
            - For bench == 'termination', remove a trailing '.hoa' from the file name
                (files are of the form '...ba.hoa') and replace any path directory
                named 'ba' with 'hoa'.
            - For bench == 'autohyper', append the '.ba' extension to the file name
                (files are of the form '... .hoa') and replace directory names as
                follows: gni -> gni_ba, nusmv -> nusmv_ba, planning -> planning_ba.
            - For bench == 'pecan', remove the trailing BA extension: if the
                filename ends with '.aligned.ba' drop that entire suffix; if
                it ends with just '.ba', drop '.ba'.
      - For other benches return the original `ba_file` unchanged.

    Args:
        ba_file (str): input BA filename
        hoa_files: (unused) placeholder for potential other implementations
        bench (str): benchmark name

    Returns:
        str: filename to use as HOA equivalent
    """
    
    if ba_file is None:
        return ba_file

    # Handle the termination benchmarks
    if bench == 'termination':
        p = Path(ba_file)
        # Remove trailing ".hoa" from the filename if present (e.g., "...ba.hoa" -> "...ba")
        name = p.name
        if name.endswith('.ba'):
            name = name + ".hoa"
            p = p.with_name(name)

        # Replace any directory component exactly named "ba" with "hoa"
        parts = list(p.parts)
        new_parts = [ ('hoa' if part == 'ba' else part) for part in parts ]
        try:
            # Rebuild path from parts; this preserves relative vs absolute on POSIX
            from pathlib import PurePosixPath
            rebuilt = PurePosixPath(*new_parts)
            # Convert back to Path to keep original type/semantics
            p = Path(str(rebuilt))
        except Exception:
            # Fallback to a safe string replace on path separators
            p = Path(str(p).replace('/ba/', '/hoa/'))

        return str(p)

    # Handle the autohyper benchmarks
    if bench == 'autohyper':
        p = Path(ba_file)

        # Replace directory components accordingly (keep existing logic).
        parts = list(p.parts)
        mapping = {'gni_ba': 'gni', 'nusmv_ba': 'nusmv', 'planning_ba': 'planning'}
        new_parts = [mapping.get(part, part) for part in parts]
        try:
            from pathlib import PurePosixPath
            rebuilt = PurePosixPath(*new_parts)
            p = Path(str(rebuilt))
        except Exception:
            # Fallback simple string replacements on POSIX-style paths
            s = str(p)
            s = s.replace('/gni/', '/gni_ba/')
            s = s.replace('/nusmv/', '/nusmv_ba/')
            s = s.replace('/planning/', '/planning_ba/')
            p = Path(s)

        # For autohyper just change .ba extension to .hoa (if present)
        if p.suffix == '.ba':
            p = p.with_suffix('.hoa')

        return str(p)

    if bench == 'rabit':
        p = Path(ba_file)
        # If the file ends with '.ba' replace that suffix with '.hoa'
        if p.suffix == '.ba':
            return str(p.with_suffix('.hoa'))
        # If '.ba' appears in the filename (e.g. 'x.ba.hoa'), replace the last
        # occurrence of '.ba' with '.hoa' while preserving other path parts.
        name = p.name
        idx = name.rfind('.ba')
        if idx != -1:
            new_name = name[:idx] + '.hoa'
            return str(p.with_name(new_name))
        # Fallback: append .hoa as the suffix
        return str(p.with_suffix('.hoa'))

    # Handle the pecan benchmarks: strip trailing '.aligned.ba' or '.ba'
    if bench == 'pecan':
        p = Path(ba_file)
        name = p.name
        if name.endswith('.ba'):
            new_name = name[: -len('.ba')]
            return str(p.with_name(new_name))
        return str(p)

    # Default: no conversion
    return ba_file

def filter_inclusion_pairs_unique(df: pd.DataFrame,
                                   name_column: str = "name",
                                   prefer_patterns=None,
                                   keep_debug_cols: bool = False) -> pd.DataFrame:
    """Filter inclusion results to keep only one direction per unordered pair.

    Rows are identified by a "pair" encoded in the ``name`` column as
    "<file1>+<file2>". Datasets often contain both directions (``A+B`` and
    ``B+A``). This function keeps exactly one row per unordered pair, with a
    preference for rows where the first file matches certain filename patterns.

    Preference (from highest to lowest):
      1) *sup.autfilt.aligned
      2) *sup.autfilt
      3) *A.ba.hoa
      4) *A.hoa

    If none of the patterns match (or there's a tie), a deterministic
    tiebreaker is applied: prefer the direction where ``file1 <= file2``
    lexicographically; if still tied, the lexicographically smallest full
    ``name`` is chosen.

    Grouping scope: Filtering is performed globally (no per-benchmark grouping),
    i.e. at most one row remains per unordered pair across the whole dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing a column with pair names.
        name_column (str): Column containing pairs in the form "f1+f2". Default "name".
        prefer_patterns (list[str]|None): Optional override of preference patterns
            (ordered, highest priority first). If None, defaults to the list above.
        keep_debug_cols (bool): When True, keep helper columns used for selection.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame):
            - filtered: dataframe with at most one row per unordered pair (current behaviour)
            - excluded: dataframe containing the rows that were not kept
    """
    import fnmatch

    if name_column not in df.columns:
        # Nothing to do if there's no name column
        return df, df.iloc[0:0]

    if prefer_patterns is None:
        # Order from most to least specific to avoid overshadowing (aligned before autfilt)
        prefer_patterns = [
            "*sup.autfilt.aligned",
            "*sup.autfilt",
            "*A.ba.hoa",
            "*A.hoa",
        ]

    # Work on a copy to avoid mutating the caller's dataframe
    work = df.copy(deep=True)

    # Split name column into file1 and file2 using a robust rule:
    # choose the '+' that directly precedes the start of the second path (usually '.' or '/').
    # If no such separator is found, fall back to the last '+' in the string.
    import re

    def _split_pair_name(name: str):
        s = str(name) if name is not None else ""
        if not s:
            return ("", "")
        # Prefer '+' that is PRECEDED by an accepted file ending on the left side.
        # Accepted endings (from most specific to general):
        endings = [
            ".autfilt.aligned.ba",
            ".autfilt.aligned",
            ".autfilt.ba",
            ".autfilt",
            ".ba.hoa",
            ".hoa",
        ]
        best_idx = -1
        i = s.find("+")
        while i != -1:
            left = s[:i]
            if any(left.endswith(end) for end in endings):
                best_idx = i  # keep rightmost valid split
            i = s.find("+", i + 1)
        if best_idx != -1:
            return (s[:best_idx], s[best_idx + 1:])
        # Fallback: split at the last '+' (still better than the first)
        j = s.rfind("+")
        if j != -1:
            return (s[:j], s[j+1:])
        # No plus at all
        return (s, "")

    parts = work[name_column].apply(_split_pair_name)
    work["_file1"] = parts.map(lambda t: t[0])
    work["_file2"] = parts.map(lambda t: t[1])

    # Build an unordered key for the pair
    # Keep grouping per-benchmark when available to avoid cross-benchmark collapses
    a = work["_file1"]
    b = work["_file2"]
    ab_min = a.where(a <= b, b)
    ab_max = b.where(a <= b, a)
    work["_pair_key"] = ab_min + "||" + ab_max

    # Group key is the unordered pair only (global filtering)
    work["_group_key"] = work["_pair_key"]

    # Scoring: higher is better, based on which pattern the first file matches.
    # Additionally, prefer files that contain 'sup' over those that contain 'sub'
    # when the same pattern tier applies.
    def score_first(fname: str) -> tuple:
        tier = 0
        for idx, pat in enumerate(prefer_patterns):
            if fnmatch.fnmatch(fname, pat):
                tier = len(prefer_patterns) - idx
                break
        # Secondary bonus for 'sup' vs 'sub' in the first filename. Use contains check,
        # since full paths include these segments.
        sup_bonus = 1 if 'sup' in fname and 'sub' not in fname else 0
        sub_penalty = -1 if 'sub' in fname and 'sup' not in fname else 0
        return (tier, sup_bonus + sub_penalty)

    work["_pref_score"] = work["_file1"].map(score_first)

    # Direction preference tiebreaker (prefer lexicographically ordered direction)
    work["_dir_pref"] = (work["_file1"] <= work["_file2"]).astype(int)

    # Stable final tiebreaker on name for determinism
    work["_name_str"] = work[name_column].astype(str)

    # Within each group, select the single best row by our priority rules
    # Sort so the first row per group is the keeper
    sort_cols = ["_group_key", "_pref_score", "_dir_pref", "_name_str"]
    sort_ascending = [True, False, False, True]
    work_sorted = work.sort_values(by=sort_cols, ascending=sort_ascending, kind="mergesort")

    keep_mask = ~work_sorted["_group_key"].duplicated(keep="first")
    filtered = work_sorted.loc[keep_mask].copy()
    # Build excluded from original order for easier inspection
    excluded_indices = work_sorted.index[~keep_mask]
    excluded = work.loc[excluded_indices].copy()

    if not keep_debug_cols:
        # Drop helper cols from both
        dbg_cols = [c for c in filtered.columns if c.startswith("_")]
        if dbg_cols:
            filtered = filtered.drop(columns=dbg_cols)
        dbg_cols_ex = [c for c in excluded.columns if c.startswith("_")]
        if dbg_cols_ex:
            excluded = excluded.drop(columns=dbg_cols_ex)
        # Preserve original column order
        filtered = filtered[df.columns]
        excluded = excluded[df.columns]

    return filtered, excluded
