This is the source code for the paper "REQIBA: Regression and Deep Q-Learning for Intelligent UAV Cellular User to Base Station Association" 
by B. Galkin, E. Fonseca, R. Amer, L.A. DaSilva, and I. Dusparic, published in the IEEE Transactions on Vehicular Technology, November 2021. The code consists of several files

1. IPNNTrainingDataset.R generates the dataset that will be used to train the IPNN neural network
2. IPNNTraining.R uses this dataset to train the IPNN neural network.
3. REQIBAEvaluation.R carries out online training of the DDQN component of the REQIBA solution, and compares it to the closest distance and omni-directional SINR association, for various UAV heights
(the comparison for different densities and beamwidths is a very simple modification to the code, and is ommitted here)
4. HeuristicBenchmarks.R generates performance results for several heuristic association schemes, for benchmarking REQIBA

MiscFunctions.R includes miscellaneous functions that are used in the code.
