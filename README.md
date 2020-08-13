# mRNA Degradation Deep Learning

This is the supporting code for the mRNA degradation rate deep learning
prediction project.


## Contents

### Data

The `data` directory contains the raw data used by the project.

- `3U_sequences_final.txt` -
  Identifiers for all mRNA sequences. The file format is `<id> <sequence>`.
- `3U.models.3U.00A.seq1022_param.txt` -
  Measured degradation rates for A- 3' UTR sequences. The file format is
  `<id> <log2 degradation rate> <log2 x0> <onset time>`.
- `3U.models.3U.40A.seq1022_param.txt` -
  Measured degradation rates for A+ 3' UTR sequences. The file format is
  `<id> <log2 degradation rate> <log2 x0> <onset time>`.
- `run_linear_3U_40A_dg_BEST.out.mat` -
  Linear regression coefficients and features for A+ 3' UTR sequences,
  from Rabani et al., 2017.
- `run_linear_3U_00Am1_dg_BEST.out.mat` -
  Linear regression coefficients and features for A- 3' UTR sequences,
  from Rabani et al., 2017.
- `models_full_dg.txt` -
  Original predicted degradation rates for 3' UTR sequences, from Rabani et al., 2017.
  The file format is `<id> <log2 degratdation rate A+ > <log2 degratdation rate A- >`.
- `ss_out.txt` -
  Secondary structure information for all mRNA sequences. For each sequence
  in the dataset, the file contains three lines:
  1. The sequence itself (each T replaced with a U).
  2. Secondary structure in dot-bracket notation, and the free energy.
  3. Probability of the secondary structure (extended dot-bracket notation).
- `5U.seq0119.txt` -
  Measured degradation rates for 5' UTR mRNA sequences. These are the same sequences
  as in the main dataset described above. The file format is
  `<id> <degratdation rate A- > <degratdation rate A+ >`.

The functions in `preprocessing.py` handle reading the files above into pandas
DataFrames.

### Linear Regression

The `linear_regression.py` file contains a class that can apply the pre-trained
3' UTR linear model to a list of genetic sequences. The `Linear Regression.ipynb`
Jupyter notebook reproduces the result of the linear model on the complete dataset,
and compares them against the results from the original study. The notebook also
computes the MSE and R^2 values against the measured degradation rates.

### Neural Networks

The `neural_net.py` file contains the implementations of the various neural networks
used in the project.

The `neural_net_estimators.py` file contains adapters for said neural networks,
to interoperate with the scikit-learn API. These adapters will automatically use CUDA
if PyTorch supports it on the host system.

The `5UTR.ipynb` Jupyter notebook contains an evaluation of the ResidualDegrade
neural network on the 5' UTR dataset.

### Model Selection & Evaluation

The `model_selection.py` script provides a generic method for selecting optimal
regression model parameters, as well as evaluating regression models, on the 3' UTR
dataset.

Note: currently, evaluation of neural networks will automatically use CUDA
if PyTorch supports it on the host system.

For instance, to evaluate various parameters for the ResidualDegrade network,
on the A- degradation rates, one would run:

```
python model_selection.py
--deg-model a-
--model residual
--epochs 5
--jobs 3
--folds 3
--dev-ratio 0.9
select
--iterations 50
--eval-epochs 5
```

This would use 3 parallel jobs (`--jobs`) to:
1. Choose 0.9 of the 3' UTR of the dataset for training (`--dev-ratio`).
2. Choose 50 random parameter combinations for the neural network (`--iterations`).
3. For each parameter combination, perform 3-fold cross-validation (`--folds`)
   on the training dataset, training the neural network for 5 epochs (`--epochs`).
4. Choose the parameter combination with the lowest error (MSE).
5. Train the network on the complete training dataset for 5 epochs (`--eval-epochs`)
   with the chosen parameters.
6. Evaluate the network on the remaining 0.1 of the dataset, and report the MSE.

To evaluate the A+ regression forest (the regression forest using the A+ linear model's
features) on the complete A+ dataset one would run:

```
python model_selection.py
--deg-model a+
--model forest_lin_a_plus
--jobs 4
--folds 10
--dev-ratio 1
eval
```

This would use 4 parallel jobs (`--jobs`) to perform 10-fold cross-validation
(`--folds`) on the complete dataset (`--dev-ratio`), and report the MSE and R^2
values for all folds, in addition to the mean MSE and R^2.

The script relies on the model descriptions in the `evaluation` directory. Indeed,
the `--model` parameter specifies the module in the `evaluation` directory to load.
See the files in this directory for examples on their correct definition.

### Misc.

The `util.py` module contains various utility functions, such as creating a matrix
of matching parentheses in a given string (used for creating an adjacency matrix
for a secondary structure).


## Installation

This project requires a 64-bit installation of Python 3.7.

### Windows & Linux

All the required packages for the project are listed in the `requirements.txt` file.
To install, create a virtualenv and execute `pip install -r requirements.txt` within it.

Note that, by default, this will install the CPU-only version of PyTorch. To install
a CUDA-capable version (in order to speed-up NN training), see [the PyTorch website][1].

### macOS

macOS requires special handling to install PyTorch. Remove the lines pertaining
to PyTorch from the `requirements.txt` file, and follow the instructions
on [the official website][1].


[1]: https://pytorch.org/get-started/locally/
