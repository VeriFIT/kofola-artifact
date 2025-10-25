#!/bin/bash

# TACAS Artifact Full Benchmark Script
# This script runs all complement and inclusion benchmarks

set -e

# Default number of CPUs to use
CPUS=4

# Parse command line arguments
while getopts "j:h" option; do
    case $option in
        j)
            CPUS=$OPTARG
            ;;
        h)
            echo "Usage: $0 [-j CPUS]"
            echo ""
            echo "Options:"
            echo "  -j N      Number of CPUs to use (default=4)"
            echo "  -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Starting INCLUSION Full Benchmark Run"
echo "Using $CPUS CPUs"
echo "=========================================="
echo ""

# Activate the virtual environment
source .venv/bin/activate


# Store the original directory
ORIG_DIR=$(pwd)

# Change to benchmark directory
cd "ba-compl-eval/bench"

# Ensure tasks_names.txt is empty before starting the benchmarks (clear leftovers)
> tasks_names.txt

# Tools for inclusion: kofola, spot, spot-forq, forklift, rabit, bait
INCL_TOOLS=("kofola" "spot" "spot-forq" "forklift" "rabit" "bait")

for tool in "${INCL_TOOLS[@]}"; do
    echo "Running inclusion with tool: $tool"
    ./run_bench_incl.sh -t "$tool" -j "$CPUS"
    echo ""
done

echo ""
echo "Processing inclusion results..."
if [ -f "tasks_names.txt" ]; then
    # Change to eval directory to run get_local_task scripts
    cd ../eval
    while IFS= read -r task_file; do
        if [ -n "$task_file" ]; then
            echo "Getting local task: $task_file"
            ./get_local_task_incl.sh "$task_file"
        fi
    done < "../bench/tasks_names.txt"
    # Return to bench directory
    cd ../bench
    # Clear the tasks_names.txt file
    > tasks_names.txt
    echo "Inclusion results processed and tasks_names.txt cleared"
else
    echo "No tasks_names.txt file found"
fi
echo ""

# After finishing experiments: run the summary script in eval and print its output
if [ -d "../eval" ]; then
    echo "Running evaluation summary script: show_incl_results.py"
    cd ../eval
    # Run the Python script and capture its output (and errors)
    if command -v python3 >/dev/null 2>&1; then
        python3 show_incl_results.py || true
    else
        echo "python3 not found; skipping show_incl_results.py"
    fi
    # Return to bench directory
    cd ../bench
else
    echo "Eval directory not found; skipping evaluation summary"
fi


# Return to original directory
cd "$ORIG_DIR"

echo "=========================================="
echo "Full INCLUSION Benchmark Run Completed"
echo "=========================================="
