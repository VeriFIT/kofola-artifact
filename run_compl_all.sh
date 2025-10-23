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
echo "Starting COMPLEMENT Full Benchmark Run"
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

# Tools for complementation: kofola, kofola-subs-tup, ranker, spot
COMPL_TOOLS=("kofola" "kofola-subs-tup" "ranker" "spot" "kofola-tacas23")

for tool in "${COMPL_TOOLS[@]}"; do
    echo "Running complementation with tool: $tool"
    ./run_bench_compl.sh -t "$tool" -j "$CPUS"
    echo ""
done

echo ""
echo "Processing complementation results..."
if [ -f "tasks_names.txt" ]; then
    # Change to eval directory to run get_local_task scripts
    cd ../eval
    while IFS= read -r task_file; do
        if [ -n "$task_file" ]; then
            echo "Getting local task: $task_file"
            ./get_local_task_compl.sh "$task_file"
        fi
    done < "../bench/tasks_names.txt"
    # Return to bench directory
    cd ../bench
    # Clear the tasks_names.txt file
    > tasks_names.txt
    echo "Complementation results processed and tasks_names.txt cleared"
else
    echo "No tasks_names.txt file found"
fi
echo ""

# Return to original directory
cd "$ORIG_DIR"

echo "=========================================="
echo "Full COMPLEMENT Benchmark Run Completed"
echo "=========================================="
