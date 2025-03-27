# BurTorch: Reproduce Experiments

----

The goal of this document is to describe how to prepare the environment for launch experiments.

----

# Reliable Measurements

1. Please use provided scripts to prepare your environment from `experiments/utils/prepare`.

2. You need to fix the clock rate of the CPU during experiments.

3. Please don't launch extra compute works in your machine.

4. Please turn off the Network Interface Cards.

5. Please launch experiments in the second CPU Core because the first CPU core is very often used by the system to handle interruptions in the kernel of the OS.

# Experiments

The baseline scripts for the experiments are presented in the table below. Please launch the corresponding scripts from the corresponding folder, which contains the scripts themselves. For compiled baselines, we provide short build scripts that you can use to build and run experiments without any IDE. In addition, `save` experiments produce binary files used for `load` experiments. 


| Number | Experiment Name | Path to Baselines | Path to BurTorch |
|--------|---------------|-------------------|------------------|
| 1 | Tiny graph example | [experiments/exp1_tiny_example](./experiments/exp1_tiny_example) | [burt/bin_tiny_example](./burt/bin_tiny_example) |
| 2 | Small graph example | [experiments/exp2_small_example](./experiments/exp2_small_example) | [burt/bin_small_example](./burt/bin_small_example) |
| 3 | Small graph example Loading | [experiments/exp3_small_example_loading](./experiments/exp3_small_example_loading) | [burt/bin_small_example_loading](./burt/bin_small_example_loading) |
| 4 | Small example saving | [experiments/exp4_small_example_saving](./experiments/exp4_small_example_saving) | [burt/bin_small_example_saving](./burt/bin_small_example_saving) |
| 5 | Character-level autoregressive prediction based on the MLP model. | [experiments/exp5_makemore_nlp_example](./experiments/exp5_makemore_nlp_example) | [burt/bin_makemore_nlp_example](./burt/bin_makemore_nlp_example) (requires [datasets/names.txt](./datasets/names.txt) as input) |
| 6 | Character-level autoregressive prediction based on GPT-3-like model. | [experiments/exp6_gpt3_example](./experiments/exp6_gpt3_example) | [burt/bin_gpt_example](./burt/bin_gpt_example) (requires [datasets/input.txt](./datasets/input.txt) as input) |
| 7 | GPT-3 Example | [experiments/exp7_small_example_energy](./experiments/exp7_small_example_energy) | [burt/bin_small_example_for_energy](./burt/bin_small_example_for_energy) |

# Utils

* [compute](./experiments/utils/compute): Help scripts make computations with humans in the loop.
* [measure]./experiments/utils/(measure): A small binary that measures the execution time of the process under Windows OS. 
* [prepare](./experiments/utils/prepare): Help scripts to prepare environments for reliable measurements.
* [plotgraph](./experiments/utils/plotgraph) Helped the script visualize graphs from text descriptions in dot format via invoking a dot binary application from Graphviz.
* [plotenergy](./experiments/utils/plotenergy): Help scripts to prepare visualization for consumed energy.
