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

# BaseLines

The baseline scripts for the experiments presented in the paper are available here:

* experiments/exp1_tiny_example
* experiments/exp2_small_example
* experiments/exp3_small_example_loading
* experiments/exp4_small_example_saving
* experiments/exp5_makemore_nlp_example
* experiments/exp6_gpt3_example

Please launch correspond scripts from the corresponded folder which contains the scripts itself.

# BurTorch:

* burt/bin_tiny_example - experiment 1. tiny example benchmark and in addition saving serialized graph repreresentation. To visualize graph use `experiments\utils\plotgraph`.
* burt/bin_small_example - experiment 2. small example with computation graph from `https://github.com/karpathy/micrograd?tab=readme-ov-file#example-usage`. To visualize graph use `experiments\utils\plotgraph`.
* burt/bin_small_example_loading - experiment 3. loading small compute graph. Please use binary files from next experiment 4.
* burt/bin_small_example_saving - experiment 4. saving small compute graph.
* burt/bin_makemore_example - experiment 5. please provide path to `datasets/names.txt`
* burt/bin_gpt_example - experiment 6. please provide path to `datasets/input.txt` file.

