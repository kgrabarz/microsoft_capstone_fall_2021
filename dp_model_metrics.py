import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from diffprivlib.models import LogisticRegression as DPLR
# from diffprivlib.models import RandomForestClassifier as DPRF


## DP MODEL CLASSIFICATION ##

def dp_model_classification_helper(eps, classifier, train_df, test_df, dataset="adult"):
    """ helper function to calculate TPR_diff, FPR_diff, and F1_score for a DP classifier at a given epsilon """

    # Get name of target and protected attribute
    if dataset == "adult":
        target = "label"
        protected_att = "sex"
    elif dataset == "acs":
        pass
    elif dataset == "compas":
        target = "two_year_recid"
        protected_att = "race"
    elif dataset == "german":
        target = "credit_risk"
        protected_att = "age"
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Check dimensions of dfs (for one-hot encoded data)
    if len(train_df.columns) < len(test_df.columns):
        missing_cols = list(set(test_df.columns) - set(train_df.columns))
        # Add missing columns to train_df
        for col in missing_cols:
            train_df[col] = 0
        # Reorder columns to match test_df
        train_df = train_df[test_df.columns]
        print("Columns added to train_df. This should only happen if you are passing in one-hot encoded data.")
    elif len(train_df.columns) == len(test_df.columns):
        pass
    else:
        raise Exception("train_df has more columns than test_df.")

    # Train the classification model
    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_test, y_test = test_df.drop(target, axis=1), test_df[target]
    if classifier == "logistic":
        m = DPLR(epsilon=eps, max_iter=1000)
        m.fit(X_train, y_train)
    elif classifier == "forest":
        m = DPRF(epsilon=eps, random_state=0)
        m.fit(X_train, y_train)
    else:
        raise Exception(f"Classifier {classifier} not recognized.")

    # Predict on the test set
    y_test_pred = m.predict(X_test)

    # Get the all classification metrics on the test set
    df = X_test.copy()
    df["y"] = y_test
    df["y_pred"] = y_test_pred
    TP_f = len(df[(df[protected_att]==0) & (df["y"]==1) & (df["y_pred"]==1)])
    TP_m = len(df[(df[protected_att]==1) & (df["y"]==1) & (df["y_pred"]==1)])
    FP_f = len(df[(df[protected_att]==0) & (df["y"]==0) & (df["y_pred"]==1)])
    FP_m = len(df[(df[protected_att]==1) & (df["y"]==0) & (df["y_pred"]==1)])
    TN_f = len(df[(df[protected_att]==0) & (df["y"]==0) & (df["y_pred"]==0)])
    TN_m = len(df[(df[protected_att]==1) & (df["y"]==0) & (df["y_pred"]==0)])
    FN_f = len(df[(df[protected_att]==0) & (df["y"]==1) & (df["y_pred"]==0)])
    FN_m = len(df[(df[protected_att]==1) & (df["y"]==1) & (df["y_pred"]==0)])
    TPR_f = TP_f / (TP_f + FN_f)
    TPR_m = TP_m / (TP_m + FN_m)
    FPR_f = FP_f / (FP_f + TN_f)
    FPR_m = FP_m / (FP_m + TN_m)
    TPR_diff = TPR_m - TPR_f
    FPR_diff = FPR_m - FPR_f
    f1_score = (TP_f+TP_m) / (TP_f+TP_m + 0.5*(FP_f+FP_m + FN_f+FN_m))

    return (TPR_diff, FPR_diff, f1_score)

def get_dp_model_table_metrics(epsilon_list, nreps, classifier, train_df, test_df, dataset="adult"):

    # Initialize lists
    tpr_diff_list = []
    fpr_diff_list = []
    f1_scores = []

    # Loop through the epsilon values and repetitions
    for eps in epsilon_list:
        for rep in range(nreps):

            # Get the classification metrics
            results = dp_model_classification_helper(eps=eps, classifier=classifier, train_df=train_df,
                                                     test_df=test_df, dataset=dataset)
            if results is not None:
                TPR_diff, FPR_diff, f1_score = results

                # Append metrics to lists
                tpr_diff_list.append(TPR_diff)
                fpr_diff_list.append(FPR_diff)
                f1_scores.append(f1_score)

    # Get medians
    tpr_diff_median = np.median(tpr_diff_list)
    fpr_diff_median = np.median(fpr_diff_list)
    f1_score_metric = np.median(f1_scores)

    return tpr_diff_median, fpr_diff_median, f1_score_metric
