## Kofola 1.0: A Modular Approach to ω-Regular Complementation and Inclusion Checking

This artifact reproduces the experiments for the paper above.
We assume that you downloaded the artifact archive, extracted it in your \$HOME inside the [TACAS 2026 Artifact Evaluation Virtual Machine](https://zenodo.org/records/17171929) with basic build tools pre-installed (Ubuntu 25.04).
No network connection should be necessary.

- Applying for badges: *Available, Functional, Reusable*
- Link to the artifact: https://doi.org/10.5281/zenodo.17457593
- Link to the artifact repository: https://github.com/VeriFIT/kofola-artifact
- Link to the tool repository: https://github.com/VeriFIT/kofola
- Platform: amd64

### Hardware requirements

- For the evaluation with default parameters, it is necessary to have at least 4 CPU cores (parallelism is configurable) and 32 GB of RAM memory. 
- In order  to reproduce the results reported in the paper you should allocate about 8 GiB of memory per CPU
- If you have fewer resources, you can reduce parallelism with the `-j` parameter (see below).
- Tested on 5 cores amd64 CPU with 64 GB of RAM memory (the estimated running times below are computed wrt this configuration).

### Installation (~30 min)

Extract and install from HOME:

```bash
cd ~/kofola-artifact
./install.sh
```

Notes:
- Requires sudo and internet access (builds and install all necessary tools, installs Python deps in .venv/).
- You might be required to fill sudo password multiple times.

### Running smoke test (~1 min)

Quickly verifies the installation on a small subset of complementation experiments (evaluates all tools on S1S benchmark).

```bash
cd ~/kofola-artifact
./run_smoke.sh
```

Parameters for `run_smoke.sh`:
- `-j N`  Number of CPUs to use (default: 4). Reduce `-j` if you have fewer CPUs or less memory.

You are supposed to see something like the following summary table (fill in the measured values):

| Tool | Unsolved | Avg | Unsupported |
|------|---------:|----:|------------:|
| Kofola | 0 | 4.65946 | 0 |
| Spot | 0 | 4.89189 | 0 |
| KofolaOld | 0 | 4.58108 | 0 |
| KofolaSlice | 0 | 4.67568 | 0 |
| Ranker | 1 | 8.729 | 0 |

Moreover, in `~/kofola-artifact/ba-compl-eval/eval/` you should see generated scatter plots (comparing kofola with other tools).

### Running full COMPLEMENTATION experimental suite (~4.5 hours)

For reproducing the full results from the paper concerning the complementation, use the following commands.

```bash
cd ~/kofola-artifact
./run_compl_all.sh
```

Parameters for `run_compl_all.sh`:
- `-j N`  Number of CPUs to use (default: 4). Reduce `-j` if you have fewer CPUs or less memory.

After finishing the `run_compl_all.sh` you should see results corresponding to Table 1 in the paper and scatter plots from Fig. 2.
Scatter plots are located in `~/kofola-artifact/ba-compl-eval/eval/`. Note that the numbers might be a bit different.

### Running full INCLUSION experimental suite (~2.6 hours)

For reproducing the full results from the paper concerning the inclusion testing, use the following commands.

```bash
cd ~/kofola-artifact
./run_incl_all.sh
```

Parameters for `run_incl_all.sh`:
- `-j N`  Number of CPUs to use (default: 4). Reduce `-j` if you have fewer CPUs or less memory.

After finishing the `run_incl_all.sh` you should see results corresponding to Table 2 in the paper and scatter plots from Fig. 3.
Scatter plots are located in `~/kofola-artifact/ba-compl-eval/eval/`. Note that the times and the number of solved instances might be a bit different, but the overall trend should stay the same.

### Regenerating table results and scatter plots

After running the experiments, you can regenerate the summary tables and scatter plots from the collected results.

1) Activate the Python virtual environment:

```bash
source kofola-artifact/.venv/bin/activate
```

2) Complementation results (tables + scatter plots):

```bash
cd ~/kofola-artifact/ba-compl-eval/eval
python3 show_compl_results.py
```

3) Inclusion results (tables + scatter plots):

```bash
cd ~/kofola-artifact/ba-compl-eval/eval
python3 show_incl_results.py
```

- Both scripts print formatted tables that you could see after finishing the experimental evaluation (by running`run_compl_all.sh` and `run_incl_all.sh`). Moreover, it re-generates scatter plots to the folder `~/kofola-artifact/ba-compl-eval/eval/`.

### Reusability
Below are quick guidelines to extend the artifact with new tools or new benchmark sets.

#### Kofola sources
After installation, the source code of Kofola is located at `ba-compl-eval/bin/kofola`. The README file in Kofola's root folder contains detailed information about the tool.

New partial complementation procedures can be added to Kofola into the folder `src/algorithms` (see some of the files there&mdash;e.g., `complement_alg_mh.hpp` and `complement_alg_mh.cpp`&mdash;for inspiration).
You also need to modify the `select_algorithms()` method in `src/complement/complement_sync.cpp` to pick the new partial algorithm for corresponding component types (again, see how this is done for other algorithms).

#### Adding a new tool

- Provide a wrapper script that standardizes how the tool is called by the benchmark runner.
	- Location: `ba-compl-eval/bin/your-tool-wrap.sh`
	- Use existing wrappers as templates (e.g., `ba-compl-eval/bin/spot-wrap.sh`, `ba-compl-eval/bin/kofola-wrap.sh`, `ba-compl-eval/bin/rabit-wrap.sh`).
	- Make sure the script:
		- Parses inputs consistently with other wrappers (paths to automata, parameters).
		- Returns appropriate exit codes and produces the expected outputs so the evaluator can parse results.
		- Is executable (`chmod +x ba-compl-eval/bin/your-tool-wrap.sh`).

- Register the tool in the benchmark configuration so it’s picked up by the runners:
	- Complementation: add it under `tool:` in `ba-compl-eval/bench/omega-compl.yaml`.
	- Inclusion: add it under `tool:` in `ba-compl-eval/bench/omega-incl.yaml`.
	- Point the command to your wrapper script and set any tool-specific parameters.

- For running the experiments on particular benchmarks, use `bench/run_bench_compl.sh` or `bench/run_bench_incl.sh`.

#### Adding new benchmarks

- Create a new `.input` file listing the automata to evaluate (one path per line).
	- Locations:
		- Complementation inputs: `ba-compl-eval/bench/inputs/compl/your-benchmark.input`
		- Inclusion inputs: `ba-compl-eval/bench/inputs/incl/your-benchmark.input`
	- Contents: plain text with paths to automata files. See existing examples like `ba-compl-eval/bench/inputs/compl/s1s.input`, `ba-compl-eval/bench/inputs/compl/seminator.input`, or `ba-compl-eval/bench/inputs/incl/rabit.input`.
	- Typical sources live under `ba-compl-eval/automata/automata-benchmarks/...`, but you can reference your own files as well.

- Ensure your new input is referenced by the benchmark configuration as needed:
	- The full-suite runners (`run_compl_all.sh`, `run_incl_all.sh`) use the inputs configured in `omega-compl.yaml` and `omega-incl.yaml`.
	- To run only your new set, you can temporarily adjust the corresponding YAML to include just your `.input`, or follow the patterns in `ba-compl-eval/bench/run_bench_compl.sh` / `ba-compl-eval/bench/run_bench_incl.sh`.
