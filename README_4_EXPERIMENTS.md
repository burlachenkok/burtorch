# BurTorch: Reproduce Experiments

----

The goal of this document is to describe how after preparing the environment launch experiments.

----

# Reliable Measurements

1. Please use provided scripts to prepare your environment from `experiments/utils/prepare`.

2. You need to fix the clock rate of CPU during experiments.

3. Please don't launch extra compute works in your machine.

4. Please turn of the Network Interface Cards.

5. Please launch experiments in the second CPU Core, because first CPU core very often use by system to handle interrupts in the kernel of OS.

# Experiments

The baseline scripts for the experiments presented in the table below. Please launch correspond scripts from the corresponded folder which contains the scripts itself. For compiled baselines we provide short build scripts which you can use to build and run experiments wihout any IDE. In addition for `save` experiments produce binary files used for `load` experiments. 


| Number | Experiment Name | Path to Baselines | Path to BurTorch |
|--------|---------------|-------------------|------------------|
| 1 | Tiny graph example | `experiments/exp1_tiny_example` | `burt/bin_tiny_example` |
| 2 | Small graph example | `experiments/exp2_small_example` | `burt/bin_small_example` |
| 3 | Small graph example Loading | `experiments/exp3_small_example_loading` | `burt/bin_small_example_loading` |
| 4 | Small example saving | `experiments/exp4_small_example_saving` | `burt/bin_small_example_saving` |
| 5 | Character-level autoregressive prediction based on MLP model. | `experiments/exp5_makemore_nlp_example` | `burt/bin_makemore_example` (requires `datasets/names.txt` as input) |
| 6 | Character-level autoregressive prediction based on GPT-3-like model. | `experiments/exp6_gpt3_example` | `burt/bin_gpt_example` (requires `datasets/input.txt` as input) |
| 7 | Experiment 6: GPT-3 Example | `experiments/exp7_small_example_energy` | `burt/bin_small_example_for_energy` |



# Utils

* compute. Help scripts to make computation for computation with human in the loop.
* measure. Small binary which measures execution time of process under Windows OS. 
* prepare. Help scripts to prepare environments for reliable measurements.
* plotgraph. Help script to visialize graph from text description in dot format via invoking dot binary application from Graphviz.
* plotenergy. Help scripts to prepare visualization for consumed energy.
