import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from snsynth.pytorch.nn import DPCTGAN, PATECTGAN
## BINARY CLASSIFICATION ##

def get_classification_summary(train_df, test_df, classifier="logistic", evaluate="test", dataset="adult"):

    # Get name of target and protected attribute
    if dataset == "adult":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "acs":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "compas":
        target = "two_year_recid"
        protected_att = "race"
        priv_class = "White"
        unpriv_class = "Black"
    elif dataset == "german":
        target = "credit_risk"
        protected_att = "age"
        priv_class = ">25"
        unpriv_class = "<=25"
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
    elif len(train_df.columns) == len(test_df.columns):
        pass
    else:
        raise Exception("train_df has more columns than test_df.")

    # Train the classification model
    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_test, y_test = test_df.drop(target, axis=1), test_df[target]
    if classifier == "logistic":
        m = LogisticRegression(max_iter=1000, C=1., penalty="l2")
        m.fit(X_train, y_train)
    elif classifier == "forest":
        m = RandomForestClassifier(random_state=0)
        m.fit(X_train, y_train)
    else:
        raise Exception(f"Classifier {classifier} not recognized.")

    # Predict on the train and test data
    y_train_pred = m.predict(X_train)
    y_test_pred = m.predict(X_test)

    # Get the all classification metrics
    if evaluate == "test":
        df = X_test.copy()
        df["y"] = y_test
        df["y_pred"] = y_test_pred
    else:
        df = X_train.copy()
        df["y"] = y_train
        df["y_pred"] = y_train_pred
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
    ACC_f = (TP_f + TN_f) / sum([TP_f, FP_f, TN_f, FN_f])
    ACC_m = (TP_m + TN_m) / sum([TP_m, FP_m, TN_m, FN_m])
    F1_score = (TP_f+TP_m) / (TP_f+TP_m + 0.5*(FP_f+FP_m + FN_f+FN_m))

    # Print out results
    print(f"\nCLASSIFICATION RESULTS ({classifier}, eval on test data)\n")
    print("True positive rates:")
    print(f"{unpriv_class}: {TPR_f:.4f}, {priv_class}: {TPR_m:.4f}\n")
    print("False positive rates:")
    print(f"{unpriv_class}: {FPR_f:.4f}, {priv_class}: {FPR_m:.4f}\n")
    print("Equalized odds distances:")
    print(f"y=1: {TPR_diff:.4f}, y=0: {FPR_diff:.4f}\n")
    print("Classification accuracies:")
    print(f"{unpriv_class}: {ACC_f:.4f}, {priv_class}: {ACC_m:.4f}\n")
    print(f"F1-score: {F1_score:.4f}\n")

    return y_train_pred, y_test_pred

def classification_helper(synthesizer, eps, rep, classifier, test_df, non_priv_train=None, one_hot_encode_train=False, dataset="adult", results_dir=""):
    """ helper function to calculate TPR_diff, FPR_diff, and F1_score for a single synthesizer, eps, rep, and classifier """

    # Get name of target and protected attribute
    if dataset == "adult":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "acs":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "compas":
        target = "two_year_recid"
        protected_att = "race"
        priv_class = "White"
        unpriv_class = "Black"
    elif dataset == "german":
        target = "credit_risk"
        protected_att = "age"
        priv_class = ">25"
        unpriv_class = "<=25"
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Read in the synthetic training data or use non-private data
    if non_priv_train is not None:
        train_df = non_priv_train
    else:
        fname = results_dir+f"{synthesizer}_eps={eps}_rep={rep}.csv"
        train_df = pd.read_csv(fname, index_col=0)

    # One-hot encode the training data if specified
    if one_hot_encode_train:
        train_df = one_hot_encode(train_df, dataset=dataset)

    # Check dimensions of dfs (for one-hot encoded data)
    if len(train_df.columns) < len(test_df.columns):
        missing_cols = list(set(test_df.columns) - set(train_df.columns))
        # Add missing columns to train_df
        for col in missing_cols:
            train_df[col] = 0
        # Reorder columns to match test_df
        train_df = train_df[test_df.columns]
    elif len(train_df.columns) == len(test_df.columns):
        pass
    else:
        raise Exception("train_df has more columns than test_df.")

    # Train the classification model
    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_test, y_test = test_df.drop(target, axis=1), test_df[target]
    if len(np.unique(y_train)) != 2:
        return None
    if classifier == "logistic":
        m = LogisticRegression(max_iter=1000, C=1., penalty="l2")
        m.fit(X_train, y_train)
    elif classifier == "forest":
        m = RandomForestClassifier(random_state=0)
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

def get_table_metrics(synthesizer, epsilon_list, nreps, classifier, test_df, f1_metric="median", one_hot_encode_train=False, dataset="adult", results_dir=""):
    """ return median TPR_diff, FPR_diff, F1_score across all epsilons and repetitions """

    # Initialize lists
    tpr_diff_list = []
    fpr_diff_list = []
    f1_scores = []

    # Loop through the epsilon values and repetitions
    for eps in epsilon_list:
        for rep in range(nreps):

            # Get the classification metrics
            results = classification_helper(synthesizer=synthesizer, eps=eps, rep=rep, classifier=classifier,
                                            test_df=test_df, one_hot_encode_train=one_hot_encode_train, dataset=dataset, results_dir=results_dir)
            if results is not None:
                TPR_diff, FPR_diff, f1_score = results

                # Append metrics to lists
                tpr_diff_list.append(TPR_diff)
                fpr_diff_list.append(FPR_diff)
                f1_scores.append(f1_score)

    # Get medians
    tpr_diff_median = np.median(tpr_diff_list)
    fpr_diff_median = np.median(fpr_diff_list)
    if f1_metric == "median":
        f1_score_metric = np.median(f1_scores)
    elif f1_metric == "max":
        f1_score_metric = np.max(f1_scores)

    return tpr_diff_median, fpr_diff_median, f1_score_metric

def get_plot_metrics(synthesizer, epsilon_list, nreps, classifier, test_df, f1_metric="median", one_hot_encode_train=False, dataset="adult", results_dir=""):
    """ return median and std of TPR_diff, FPR_diff, F1_score for *each* epsilon value (arrays of values) """

    # Initialize lists
    tpr_diff_median_list = []
    fpr_diff_median_list = []
    f1_score_metrics = []
    tpr_diff_std_list = []
    fpr_diff_std_list = []
    f1_score_std_list = []

    # Loop through the epsilon values
    for eps in epsilon_list:

        # Initialize list to hold values for each epsilon value
        tpr_diff_list = []
        fpr_diff_list = []
        f1_scores = []

        # Loop through the range of repetitions
        for rep in range(nreps):

            # Get the classification metrics
            results = classification_helper(synthesizer=synthesizer, eps=eps, rep=rep, classifier=classifier,
                                            test_df=test_df, one_hot_encode_train=one_hot_encode_train, dataset=dataset, results_dir=results_dir)
            if results is not None:
                TPR_diff, FPR_diff, f1_score = results

                # Append metrics to lists
                tpr_diff_list.append(TPR_diff)
                fpr_diff_list.append(FPR_diff)
                f1_scores.append(f1_score)

        # Append medians to lists
        tpr_diff_median_list.append(np.median(tpr_diff_list))
        fpr_diff_median_list.append(np.median(fpr_diff_list))
        if f1_metric == "median":
            f1_score_metrics.append(np.median(f1_scores))
        elif f1_metric == "max":
            f1_score_metrics.append(np.max(f1_scores))
        tpr_diff_std_list.append(np.std(tpr_diff_list))
        fpr_diff_std_list.append(np.std(fpr_diff_list))
        f1_score_std_list.append(np.std(f1_scores))

    return tpr_diff_median_list, fpr_diff_median_list, f1_score_metrics, tpr_diff_std_list, fpr_diff_std_list, f1_score_std_list

def get_epsilon_plots(synthesizer_list, epsilon_list, nreps, classifier, test_df, f1_metric="median", non_priv_train=None, one_hot_encode_train=False, dataset="adult", results_dir="", savefig=False):
    """ return subplot with three plots showing graphs of TPR_diff, FPR_diff, F1_score acros epsilons for each synthesizer """

    # Get name for plotting
    if dataset == "adult":
        title = "Adult"
    elif dataset == "acs":
        title = "ACS Income"
    elif dataset == "compas":
        title = "COMPAS"
    elif dataset == "german":
        title = "German Credit"
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Initialize subplots
    if savefig:
        fig, ax = plt.subplots(1, 3, figsize=(16,3.7))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(16,5))

    # Add the non-private results to the plots
    if non_priv_train is not None:
        non_priv_tpr_diff, non_priv_fpr_diff, non_priv_f1_score = classification_helper(None, None, None, classifier=classifier, test_df=test_df, non_priv_train=non_priv_train, one_hot_encode_train=one_hot_encode_train, dataset=dataset)
        ax[0].hlines(non_priv_tpr_diff, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data", color="black")
        ax[1].hlines(non_priv_fpr_diff, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data", color="black")
        ax[2].hlines(non_priv_f1_score, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data", color="black")

    # Loop through the synthesizers
    for synth in synthesizer_list:

        # Get all classification metrics
        if synth == "QUAIL_PATECTGAN":
            epsilon_list = epsilon_list[1:]
        tpr_diff_median_list, fpr_diff_median_list, f1_score_metrics, tpr_diff_std_list, fpr_diff_std_list, f1_score_std_list \
        =  get_plot_metrics(synthesizer=synth, epsilon_list=epsilon_list, nreps=nreps, classifier=classifier, test_df=test_df, f1_metric=f1_metric,
                            one_hot_encode_train=one_hot_encode_train, dataset=dataset, results_dir=results_dir)

        # Plot the metrics with error bars
        ax[0].errorbar(epsilon_list, tpr_diff_median_list, tpr_diff_std_list, label=synth, alpha=0.7)
        ax[1].errorbar(epsilon_list, fpr_diff_median_list, fpr_diff_std_list, label=synth, alpha=0.7)
        ax[2].errorbar(epsilon_list, f1_score_metrics, f1_score_std_list, label=synth, alpha=0.7)

    # Plotting details
    if savefig:
        for i in range(3):
            ax[i].set_xlabel("Privacy budget $\epsilon$")
            ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25),
                         fancybox=True, shadow=True, ncol=2)
        ax[0].set_ylabel("Equalized odds distance ($y=1$)")
        ax[1].set_ylabel("Equalized odds distance ($y=0$)")
        ax[2].set_ylabel("F1-score")
        plt.savefig(results_dir+f"{dataset}_{classifier}.png", dpi=300, bbox_inches="tight")
    else:
        for i in range(3):
            ax[i].set_xlabel("Privacy budget $\epsilon$")
            ax[i].legend(loc="lower left")
        ax[0].set_ylabel("Equalized odds distance ($y=1$)")
        ax[1].set_ylabel("Equalized odds distance ($y=0$)")
        ax[2].set_ylabel("F1-score")
        if classifier == "logistic":
            title = f"{title} classification summary (logistic regression)"
        elif classifier == "forest":
            title = f"{title} classification summary (random forest)"
        fig.suptitle(title, size=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
