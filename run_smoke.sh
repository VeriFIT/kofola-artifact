#!/bin/bash

# TACAS Artifact Smoke Test Script
# This script runs limited complement and inclusion benchmarks for smoke testing

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
echo "Starting Smoke Test"
echo "Using $CPUS CPUs"
echo "=========================================="
echo ""

# Activate the virtual environment
source .venv/bin/activate


# Store the original directory
ORIG_DIR=$(pwd)

# Change to benchmark directory
cd "ba-compl-eval/bench"

echo "=========================================="
echo "Running Complementation Benchmarks"
echo "Benchmark: s1s"
echo "=========================================="
echo ""

# Tools for complementation: kofola, kofola-subs-tup, cola, ranker, spot
COMPL_TOOLS=("kofola" "kofola-subs-tup" "cola" "ranker" "spot")

for tool in "${COMPL_TOOLS[@]}"; do
    echo "Running complementation with tool: $tool"
    ./run_bench_compl.sh -t "$tool" -j "$CPUS" s1s
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

echo "=========================================="
echo "Running Inclusion Benchmarks"
echo "Benchmark: rabit"
echo "=========================================="
echo ""

# Tools for inclusion: kofola, spot, spot-forq, forklift, rabit, bait
INCL_TOOLS=("kofola" "spot" "spot-forq" "forklift" "rabit" "bait")

for tool in "${INCL_TOOLS[@]}"; do
    echo "Running inclusion with tool: $tool"
    ./run_bench_incl.sh -t "$tool" -j "$CPUS" rabit
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

# Return to original directory
cd "$ORIG_DIR"

echo "=========================================="
echo "Smoke Test Completed"
echo "=========================================="