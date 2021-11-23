import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from snsynth import MWEMSynthesizer
from snsynth import QUAILSynthesizer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, PATECTGAN
from diffprivlib.models import LogisticRegression as DPLR
from diffprivlib.models import RandomForestClassifier as DPRF

## PREPROCESSING ##

def convert_to_cat(df, dataset="adult"):
    
    if dataset == "adult":

        # Remove several features
        df = df.drop(["fnlwgt", "native_country", "capital_gain", "capital_loss", "marital"], axis=1)

        # Remove missing values
        df = df[~df.eq("?").any(1)]

        # Bin the continuous columns
        age_bins = [0, 25, 35, 45, 60, np.inf]
        age_labels = ["<25", "25-35", "35-45", "45-60", "60+"]
        df["age"] = pd.cut(df["age"], age_bins, labels=age_labels)
        hours_week_bins = [0, 35, 45, 60, np.inf]
        hours_week_labels = ["<35", "35-45", "45-60", "60+"]
        df["hours_week"] = pd.cut(df["hours_week"], hours_week_bins, labels=hours_week_labels)

        # Discretize the features
        for col in df.columns:
            if col not in ["label", "sex", "education_num"]:
                df[col] = pd.factorize(df[col])[0]

        # Map binary features to 0/1
        df["sex"] = df["sex"].map({"Female":0, "Male":1})
        df["label"] = df["label"].map({"<=50K":0, ">50K":1})
        
    elif dataset == "acs":
        pass
    
    elif dataset == "compas":
        pass
    
    elif dataset == "german":
        pass
    
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    return df


## DATA SYNTHESIS ##

def get_synthesizer(synthesizer, epsilon):

    # Instantiate an MWEM synthesizer
    if synthesizer == "MWEM":
        synth = MWEMSynthesizer(epsilon=epsilon, q_count=500, iterations=30, mult_weights_iterations=15, 
        splits=[], split_factor=2, max_bin_count=400)

    elif synthesizer == "DPCTGAN":
        synth = PytorchDPSynthesizer(epsilon=epsilon, gan=DPCTGAN(), preprocessor=None)
        
    elif synthesizer == "PATECTGAN":
        synth = PytorchDPSynthesizer(epsilon=epsilon, gan=PATECTGAN(), preprocessor=None)

    return synth

def get_quail_synthesizer(synthesizer, classifier, epsilon, eps_split, target):

    # Instantiate a synthesizer
    if synthesizer == "MWEM":
        def QuailSynth(epsilon):
            return MWEMSynthesizer(epsilon=epsilon, q_count=500, iterations=30, mult_weights_iterations=15, 
                                    splits=[], split_factor=2, max_bin_count=400)
        
    elif synthesizer == "DPCTGAN":
        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(epsilon=epsilon, gan=DPCTGAN(), preprocessor=None)
        
    elif synthesizer == "PATECTGAN":
        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(epsilon=epsilon, gan=PATECTGAN(), preprocessor=None)
    
    # Instantiate a DPLR classifier
    if classifier == "DPLR":
        def QuailClassifier(epsilon):
            return DPLR(epsilon=epsilon)

    # Create a QUAIL synthesizer with base synthesizer and DP classifier
    quail = QUAILSynthesizer(epsilon, QuailSynth, QuailClassifier, target, eps_split)

    return quail

def save_synthetic_data(epsilon_vals, train_df, synthesizer="MWEM", quail=False, 
                        classifier=None, eps_split=None, n_reps=1, dataset="adult"):
    
    # Get a list of all features and target to be passed into categorical_columns
    if dataset == "adult":
        target = "label"
        all_columns = ["age", "workclass", "education", "education_num", "occupation",
                               "relationship", "race", "sex", "hours_week", "label"]
    elif dataset == "acs":
        pass
    elif dataset == "compas":
        pass
    elif dataset == "german":
        pass
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Loop through epsilon values and repetitions
    for epsilon in epsilon_vals:
        for i in range(n_reps):

            if quail:  
                # Create a QUAIL-wrapped synthesizer
                synth = get_quail_synthesizer(synthesizer, classifier, epsilon, eps_split, target=target)

                # Fit synthesizer to the training data
                synth.fit(train_df)

            else:
                # Create a regular synthesizer
                synth = get_synthesizer(synthesizer, epsilon)

                if synthesizer == "MWEM":
        
                    # Fit synthesizer to the training data
                    synth.fit(train_df.to_numpy())

                elif synthesizer in ["DPCTGAN", "PATECTGAN"]:

                  # Fit synthesizer to the training data
                  synth.fit(train_df, categorical_columns=all_columns)

            # Create and save private synthetic data
            train_synth = pd.DataFrame(synth.sample(int(train_df.shape[0])), columns=train_df.columns)
            if quail:
                train_synth.to_csv(f"QUAIL_{synthesizer}_eps={epsilon}_rep={i}.csv")
            else:
                train_synth.to_csv(f"{synthesizer}_eps={epsilon}_rep={i}.csv")
                
            print(f"Completed eps={epsilon}, rep={i+1}.")
            
def plot_distributions(df, title, dataset="adult"):
    
    # Get name of target
    if dataset == "adult":
        target = "label"
    elif dataset == "acs":
        pass
    elif dataset == "compas":
        pass
    elif dataset == "german":
        pass
    else:
        raise Exception(f"Dataset {dataset} not recognized.")
        
    # Plot distributions of all features
    fig, ax = plt.subplots(2, 5, figsize=(20,8))
    ax = ax.ravel()
    for i, col in enumerate(df.columns):
            g = sns.histplot(data=df, x=col, hue=target, multiple="stack", ax=ax[i], discrete=True)
    fig.suptitle(title, size=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
            
## BINARY CLASSIFICATION ##

def get_classification_summary(train_df, test_df, classifier="logistic", evaluate="test", dataset="adult"):
    
    # Get name of target and protected attribute
    if dataset == "adult":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "acs":
        pass
    elif dataset == "compas":
        pass
    elif dataset == "german":
        pass
    else:
        raise Exception(f"Dataset {dataset} not recognized.")
    
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

def classification_helper(synthesizer, eps, rep, classifier, test_df, non_priv_train=None, dataset="adult"):
    """ helper function to calculate TPR_diff, FPR_diff, and F1_score for a single synthesizer, eps, rep, and classifier """
    
    # Get name of target and protected attribute
    if dataset == "adult":
        target = "label"
        protected_att = "sex"
        priv_class = "Male"
        unpriv_class = "Female"
    elif dataset == "acs":
        pass
    elif dataset == "compas":
        pass
    elif dataset == "german":
        pass
    else:
        raise Exception(f"Dataset {dataset} not recognized.")
    
    # Read in the synthetic training data or use non-private data
    if non_priv_train is not None:
        train_df = non_priv_train
    else:
        fname = f"{synthesizer}_eps={eps}_rep={rep}.csv"
        train_df = pd.read_csv(fname, index_col=0)

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

def get_table_metrics(synthesizer, epsilon_list, nreps, classifier, test_df, f1_metric="median", dataset="adult"):
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
    if f1_metric == "median":
        f1_score_metric = np.median(f1_scores)
    elif f1_metric == "max":
        f1_score_metric = np.max(f1_scores)
    
    return tpr_diff_median, fpr_diff_median, f1_score_metric

def get_plot_metrics(synthesizer, epsilon_list, nreps, classifier, test_df, f1_metric="median", dataset="adult"):
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
                                            test_df=test_df, dataset=dataset)
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
        
def get_epsilon_plots(synthesizer_list, epsilon_list, nreps, classifier, test_df, f1_metric="median", non_priv_train=None, dataset="adult"):
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
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    
    # Loop through the synthesizers
    for synth in synthesizer_list:
        
        # Get all classification metrics
        tpr_diff_median_list, fpr_diff_median_list, f1_score_metrics, tpr_diff_std_list, fpr_diff_std_list, f1_score_std_list \
        =  get_plot_metrics(synth=synth, epsilon_list=epsilon_list, nreps=nreps, classifier=classifier, test_df=test_df, f1_metric=f1_metric, dataset=dataset)
        
        # Plot the metrics with error bars
        ax[0].errorbar(epsilon_list, tpr_diff_median_list, tpr_diff_std_list, label=synth)
        ax[1].errorbar(epsilon_list, fpr_diff_median_list, fpr_diff_std_list, label=synth)
        ax[2].errorbar(epsilon_list, f1_score_metrics, f1_score_std_list, label=synth)
        
    # Add the non-private results to the plots
    if non_priv_train is not None:
        non_priv_tpr_diff, non_priv_fpr_diff, non_priv_f1_score = classification_helper(None, None, None, classifier=classifier, test_df=test_df, non_priv_train=non_priv_train, dataset=dataset)
        ax[0].hlines(non_priv_tpr_diff, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data")
        ax[1].hlines(non_priv_fpr_diff, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data")
        ax[2].hlines(non_priv_f1_score, xmin=epsilon_list[0], xmax=epsilon_list[-1], linestyles="--", label="Non-private data")
        
    # Plotting details
    for i in range(3):
        ax[i].set_xlabel("Privacy budget")
        ax[i].legend()
    ax[0].set_ylabel("Equalized odds distance ($y=1$)")
    ax[1].set_ylabel("Equalized odds distance ($y=0$)")
    ax[2].set_ylabel("F1-score")
    if classifier == "logistic":
        title = f"{title} classification summary (logistic regression)"
    elif classifier == "forest":
        title = f"{title} classification summary (random forest)"
    fig.suptitle(title, size=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    
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
        pass
    elif dataset == "german":
        pass
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

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