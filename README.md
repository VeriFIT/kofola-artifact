## Kofola 1.0: A Modular Approach to ω-Regular Complementation and Inclusion Checking

This artifact reproduces the experiments for the paper above. Assume you downloaded the artifact archive, extracted it to your HOME, and you run inside a prepared Ubuntu 25.04 TACAS AE VM with basic build tools pre-installed.

### Hardware requirements

- For the evaluation with default parameters, it is necessary to have at least 4 CPU cores (parallelism is configurable) and 32 GB of RAM memory. 
- In order  to reproduce the results reported in the paper you should allocate about 8 GiB of memory per CPU
- If you have fewer resources, you can reduce parallelism with the `-j` parameter (see below).
- Tested on 5 cores CPU with 64 GB of RAM memory (the estimated running times below are computed on this configuration).

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

### Running full COMPLEMENTATION experimental suite

For reproducing the full results from the paper concerning the complementation, use the following commands.

```bash
cd ~/kofola-artifact
./run_compl_all.sh
```

Parameters for `run_compl_all.sh`:
- `-j N`  Number of CPUs to use (default: 4). Reduce `-j` if you have fewer CPUs or less memory.

After finishing the `run_compl_all.sh` you should see results corresponding to Table 1 in the paper (page 13) and scatter plots from Fig. 2 (page 14). 

### Running full INCLUSION experimental suite (~2.6 hours)

For reproducing the full results from the paper concerning the inclusion testing, use the following commands.

```bash
cd ~/kofola-artifact
./run_incl_all.sh
```

Parameters for `run_incl_all.sh`:
- `-j N`  Number of CPUs to use (default: 4). Reduce `-j` if you have fewer CPUs or less memory.

After finishing the `run_incl_all.sh` you should see results corresponding to Table 2 in the paper (page 13) and scatter plots from Fig. 3 (page 16).

### Regenerating table results and scatter plots