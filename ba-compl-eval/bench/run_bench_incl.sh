#!/bin/bash

show_help() {
	echo "Usage:"
	echo "run_bench_incl.sh [options] [BENCHMARK1 BENCHMARK2 ...]"
	echo ""
	echo "Runs the given tool on the specified benchmarks. For omega-automata"
	echo "inclusion we treat benchmarks as explicit suites (no grouping)."
	echo "If no benchmark is given, the default suites are:"
	echo "  advanced-automata, s1s-direct-red, from_ltl_red"
	echo ""
	echo "Options:"
	echo "  -h        Show this help message"
	echo "  -t TOOL   Which tool to run (default=kofola_fast)"
	echo "  -j N      How many processes to run in parallel (default=4)"
	echo "  -m N      Memory limit of each process in GB (default=8)"
	echo "  -s N      Timeout for each process in seconds (default=120)"
	
	echo "Note: positional arguments are treated as benchmark names and are not"
	echo "expanded into groups. Provide multiple benchmark names to run them all."
}

# For omega automata complementation we do not use grouped benchmarks.
# Distinguish the three benchmark suites explicitly; do not group benchmarks.
# If no benchmark is given, run the three omega-complementation benchmark sets.

tool="kofola_fast"
j_value="4"
m_value="8"
s_value="120"
while getopts "ht:j:m:s:" option; do
    case $option in
        h)
            show_help 
            exit 0
            ;;
        t)
            tool=$OPTARG
            ;;
        j)
            j_value=$OPTARG
            ;;
        m)
            m_value=$OPTARG
            ;;
        s)
            s_value=$OPTARG
            ;;
        *)
            echo "Invalid option: -$OPTARG"
            show_help
            exit 1
            ;;
    esac
done

# Shift the option index so that $1 refers to the first positional argument
shift $((OPTIND - 1))

benchmarks=()
ba_tools=("forklift" "bait" "rabit")

# If no benchmark is given, run the three omega automata complementation sets
if [ -z "$1" ]; then
  benchmarks=("autohyper" "rabit" "termination" "pecan")
else
  # treat each positional argument as a benchmark name (no grouping)
  for BENCH_NAME in "$@"; do
    benchmarks+=("$BENCH_NAME")
  done
fi

tasks_files=()

CUR_DATE=$(date +%Y-%m-%d-%H-%M)
for benchmark in "${benchmarks[@]}"; do
	echo "Running benchmark $benchmark"
	FILE_PREFIX="$benchmark-to${s_value}-$tool-$CUR_DATE"
	TASKS_FILE="$FILE_PREFIX.tasks"
    # Select input file based on whether the tool is a BA tool
    input_suffix=""
    for ba_tool in "${ba_tools[@]}"; do
        if [ "$tool" = "$ba_tool" ]; then
            input_suffix="-ba"
            break
        fi
    done
    input_file="inputs/incl/${benchmark}${input_suffix}.input"

    cat "$input_file" | ./pycobench -c omega-incl.yaml -j $j_value -t $s_value --memout $m_value -m "$tool" -o "$TASKS_FILE"
	tasks_files+=("$TASKS_FILE")
	echo "$TASKS_FILE" >> tasks_names.txt
done

for tasks_file in "${tasks_files[@]}"; do
	echo "$tasks_file"
done