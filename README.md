This directory is organized as follows:

1. data: this folder contains the graphml data that we use for our real data experiments. Sources of these data has been noted in the code and the paper

2. ELD.py: this file contains the functions that computes the ELD and creates distance matrices (it contains both the regular version and the sparse/fast version which gives speedups for large graphs (e.g. > 10000 vertices))

3. portrait_divergence.py: this file contains the functions that implements the network portrait divergence, which is our main competitor in the experiments. This file is taken directly from the github of the author of the network portrait divergence method. The links are specified in the comments.

4.
ELD Experiment (Computational Time).ipynb
ELD Experiment (Real Data - Temporal Collaboration Network and Connectomes).ipynb
ELD Experiment (Simulated, Unweighted Graphs Comparison).ipynb
ELD Experiment (Simulated, Weighted Graphs Comparison).ipynb

The above four notebooks runs the experiments that reproduces results/figures in the experiment section of the paper
Please note that for the experiments that is dependent on randomly generated graphs, depending on whether you set a random seed or not your results might vary slightly from trial to trial. 
