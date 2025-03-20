# BurTorch: Reproduce Experiments

----

The goal of this document is to describe how after preparing the environment launch experiments.

----

# Reliable Measurements

Please use provided scripts to prepare your environment from:

* `prepare folder`: The most critical thing you need to fix the clock rate of CPU during experiments.

* `use 2nd CPU core`: Please launch experiments in the second CPU Core, because first CPU core very often use by system to handle interrupts in the kernel of OS.

# BaseLines

The baseline scripts for the experiments presented in the paper are available here:

* experiments/exp1_tiny_example
* experiments/exp2_small_example
* experiments/exp3_small_example_loading
* experiments/exp4_small_example_saving
* experiments/exp5_gpt3_example

Please launch correspond scripts from the corresponded folder which contains the scripts itself.

# BurTorch:

* burt/bin_small_graph_example: no data need to be provided, it will provide information from exp1_tiny_example, exp2_small_example, exp3_small_example_loading, exp4_small_example_saving benchmarks
* burt/bin_gpt_example: please provide path to `datasets/names.txt`
* burt/bin_makemore_example: please provide path to `datasets/input.txt`
