import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


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

        # Remove unneeded features
        df = df.drop(["native_country", "Unnamed: 0"], axis=1)

        # Remove missing values
        df = df[~df.eq("?").any(1)]

        # Bin continuous columns
        age_bins = [0, 25, 35, 45, 60, np.inf]
        age_labels = ["<25", "25-35", "35-45", "45-60", "60+"]
        df["age"] = pd.cut(df["age"], age_bins, labels=age_labels)
        hours_week_bins = [0, 35, 45, 60, np.inf]
        hours_week_labels = ["<35", "35-45", "45-60", "60+"]
        df["hours_week"] = pd.cut(df["hours_week"], hours_week_bins, labels=hours_week_labels)

        # Discretize features
        for col in df.columns:
            if col not in ["label", "sex"]:
                df[col] = pd.factorize(df[col])[0]

        # Map binary features to 0/1
        df["sex"] = df["sex"].map({"Female":0, "Male":1})
        df["label"] = df["label"].map({"<=50K":0, ">50K":1})

    elif dataset == "compas":

        # Remove invalid/null entries
        df = df[(df['days_b_screening_arrest'] <= 30)
                & (df['days_b_screening_arrest'] >= -30)
                & (df['is_recid'] != -1)
                & (df['c_charge_degree'] != 'O')
                & (df['score_text'] != 'N/A')]

        # Remove races other than Caucasian or African-American
        df = df[(df['race']=='Caucasian') | (df['race']=='African-American')]

        # Calculate length_of_stay (convert into months)
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).astype(int) / 10**9
        df['length_of_stay'] /= 60 * 60 * 24 * 31
        df = df.drop_duplicates()

        # Bin the continuous columns
        age_bins = [0, 25, 35, 45, 60, np.inf]
        age_labels = ["<25", "25-35", "35-45", "45-60", "60+"]
        df["age"] = pd.cut(df["age"], age_bins, labels=age_labels)
        priors_count_bins = [-1, 3, 6, 10, 16, np.inf]
        prios_counts_labels = ["<3", "3-6", "6-10", "10-16", "16+"]
        df["priors_count"] = pd.cut(df["priors_count"], priors_count_bins, labels=prios_counts_labels)
        length_of_stay_bins = [-1, 1, 2, 5, 10, np.inf]
        length_of_stay_labels = ["<1", "1-2", "2-5", "5-10", "10+"]
        df["length_of_stay"] = pd.cut(df["length_of_stay"], length_of_stay_bins, labels=length_of_stay_labels)

        # Discretize features
        for col in df.columns:
            if col not in ["sex", "c_charge_degree", "race"]:
                df[col] = pd.factorize(df[col])[0]

        # Map binary features to 0/1
        df["sex"] = df["sex"].map({"Female":0, "Male":1})
        df["c_charge_degree"] = df["c_charge_degree"].map({"F":0, "M":1})
        df["race"] = df["race"].map({"African-American":0, "Caucasian":1})

        df = df[['age', 'priors_count', 'sex','juv_fel_count', 'juv_misd_count', 'juv_other_count',
                 'c_charge_degree', 'length_of_stay','race','two_year_recid']]

    elif dataset == "german":

        # Convert age to a binary variable
        df.loc[df["age_in_years"] <= 25, "age_in_years"] = 0
        df.loc[df["age_in_years"] > 25, "age_in_years"] = 1
        df = df.rename(columns={"age_in_years": "age"})

        # Bin the continuous columns
        df['duration_in_month'] /= 12
        df['duration_in_month'] = df['duration_in_month'].astype(int)
        df = df.rename(columns={"duration_in_month": "duration_in_year"})
        credit_amount_bins = [-1, 2000, 3500, 6500, 10000, np.inf]
        credit_amount_labels = ["<2000", "2000-3500", "3500-6500", "6500-10000", "10000+"]
        df["credit_amount"] = pd.cut(df["credit_amount"], credit_amount_bins, labels=credit_amount_labels)

        # Discretize features
        for col in ["credit_amount", "credit_history", "housing", "purpose"]:
            df[col] = pd.factorize(df[col])[0]

        df = df[['duration_in_year', 'credit_amount', 'installment_rate_in_percentage_of_disposable_income', 'age', 'credit_history', 'housing','status_of_existing_checking_account','present_employment_since', 'purpose', 'credit_risk']]

    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    return df






def one_hot_encode(cat_df, dataset="adult"):

    if dataset == "adult":
        non_binary_cols = ["age", "workclass", "education", "education_num", "occupation", "relationship", "race", "hours_week"]
    elif dataset == "acs":
        non_binary_cols = ["age", "workclass", "education", "occupation", "relationship", "race", "hours_week", "marital"]
    elif dataset == "compas":
        non_binary_cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "length_of_stay"]
    elif dataset == "german":
        non_binary_cols = ['duration_in_year', 'credit_amount', 'installment_rate_in_percentage_of_disposable_income', 'credit_history', 'housing','status_of_existing_checking_account','present_employment_since', 'purpose']
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # One-hot encode the non-binary columns
    encoded_df = pd.get_dummies(cat_df, columns=non_binary_cols, drop_first=True)

    return encoded_df
