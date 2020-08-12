# import numpy package for arrays and stuff
import numpy as np

from linear_regression import *

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from linear_regression import Model
from preprocessing import read_all_data, read_original_predictions, read_sequence_ids

# import export_graphviz
from sklearn.tree import export_graphviz

# Pre process data:
df = read_all_data("data/ss_out.txt",
                   "data/3U_sequences_final.txt",
                   "data/3U.models.3U.40A.seq1022_param.txt",
                   "data/3U.models.3U.00A.seq1022_param.txt")
df.sort_index(inplace=True)

# The DataFrame above contains truncated sequences, but we need the full ones
complete_sequences = read_sequence_ids("data/3U_sequences_final.txt")
complete_sequences.set_index("id", inplace=True)
df["sequence"] = complete_sequences["sequence"]
available_deg_rates = df.dropna()

model_a_plus = Model.load("data/run_linear_3U_40A_dg_BEST.out.mat")
model_a_minus = Model.load("data/run_linear_3U_00Am1_dg_BEST.out.mat")

# 31318 samples in total - split between train and test:
# 1264 features - a+
#  718 features - a-


# Split data to train and test.
available_deg_rates_train, available_deg_rates_test = train_test_split(available_deg_rates, test_size=0.1)

# Create kmer_cnt matrix to train and test data according to a_plus and a_minus linear regression models.
kmer_cnt_matrix_a_plus_train = model_a_plus.kmer_cnt_matrix(available_deg_rates_train["sequence"])
kmer_cnt_matrix_a_plus_test = model_a_plus.kmer_cnt_matrix(available_deg_rates_test["sequence"])

kmer_cnt_matrix_a_minus_train = model_a_minus.kmer_cnt_matrix(available_deg_rates_train["sequence"])
kmer_cnt_matrix_a_minus_test = model_a_minus.kmer_cnt_matrix(available_deg_rates_test["sequence"])

# Real deg rate values.
deg_rate_a_plus_train = available_deg_rates.loc[available_deg_rates_train.index]["log2_deg_rate_a_plus"]
deg_rate_a_plus_test = available_deg_rates.loc[available_deg_rates_test.index]["log2_deg_rate_a_plus"]
deg_rate_a_minus_train = available_deg_rates.loc[available_deg_rates_train.index]["log2_deg_rate_a_minus"]
deg_rate_a_minus_test = available_deg_rates.loc[available_deg_rates_test.index]["log2_deg_rate_a_minus"]


# create a regressor objects:
regressor_a_plus = RandomForestRegressor(n_estimators=80, random_state=None,
                                         bootstrap=True, max_depth=60, min_samples_leaf=15)

regressor_a_minus = RandomForestRegressor(n_estimators=80, random_state=None,
                                          bootstrap=True, max_depth=60, min_samples_leaf=15)

scores_a_plus = cross_val_score(
    estimator=regressor_a_plus,
    X=kmer_cnt_matrix_a_plus_train,
    y=deg_rate_a_plus_train,
    scoring="neg_mean_squared_error",
    cv=7,
    n_jobs=2,
    verbose=10,
)

print(f"Scores a plus: {scores_a_plus}")
print(f"Mean score a plus: {np.mean(scores_a_plus)}")

scores_a_minus = cross_val_score(
    estimator=regressor_a_minus,
    X=kmer_cnt_matrix_a_minus_train,
    y=deg_rate_a_minus_train,
    scoring="neg_mean_squared_error",
    cv=7,
    n_jobs=2,
    verbose=10,
)

print(f"Scores a minus: {scores_a_minus}")
print(f"Mean score a minus: {np.mean(scores_a_minus)}")


### CODE WITHOUT CROSS VALIDATION ###

# # fit the regressors:
# regressor_a_plus.fit(kmer_cnt_matrix_a_plus_train, deg_rate_a_plus_train)
# regressor_a_minus.fit(kmer_cnt_matrix_a_minus_train, deg_rate_a_minus_train)
#
# # predicting a new values:
# prediction_a_plus = regressor_a_plus.predict(kmer_cnt_matrix_a_plus_test)
# prediction_a_minus = regressor_a_minus.predict(kmer_cnt_matrix_a_minus_test)
#
# prediction_df = pd.DataFrame({"id": available_deg_rates_test.index,
#                               "a_minus": prediction_a_minus,
#                               "a_plus": prediction_a_plus})
# prediction_df.set_index("id", inplace=True)
#
# mse_a_plus = mean_squared_error(y_true=available_deg_rates_test["log2_deg_rate_a_plus"],
#                                 y_pred=prediction_df.loc[available_deg_rates_test.index]["a_plus"])
# mse_a_minus = mean_squared_error(y_true=available_deg_rates_test["log2_deg_rate_a_minus"],
#                                  y_pred=prediction_df.loc[available_deg_rates_test.index]["a_minus"])
# print(mse_a_plus)
# print(mse_a_minus)

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere, like here: http://www.webgraphviz.com/
# export_graphviz(regressor_a_minus, out_file='tree_a_minus.dot')
# export_graphviz(regressor_a_plus, out_file='tree_a_plus.dot')

