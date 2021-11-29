import numpy as np
import pandas as pd

def convert_to_str(cat_df):
    cat_df_copy = cat_df.copy()
    for col in cat_df.columns:
        cat_df_copy[col] = cat_df_copy[col].apply(str)
    return cat_df_copy

def get_spans(df, partition, scale=None):
    """
    :param        df: the dataframe for which to calculate the spans
    :param partition: the partition for which to calculate the spans
    :param     scale: if given, the spans of each column will be divided
                      by the value in `scale` for that column
    :        returns: The spans of all columns in the partition
    """
    spans = {}
    for column in df.columns:
        span = len(df[column][partition].unique())
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans

def split(df, partition, column):
    """
    :param        df: The dataframe to split
    :param partition: The partition to split
    :param    column: The column along which to split
    :        returns: A tuple containing a split of the original partition
    """
    dfp = df[column][partition]
    values = dfp.unique()
    lv = set(values[:len(values)//2])
    rv = set(values[len(values)//2:])
    return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]

def is_k_anonymous(df, partition, sensitive_column, k):
    """
    :param               df: The dataframe on which to check the partition.
    :param        partition: The partition of the dataframe to check.
    :param sensitive_column: The name of the sensitive column
    :param                k: The desired k
    :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
    """
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid, k=3):
    """
    :param               df: The dataframe to be partitioned.
    :param  feature_columns: A list of column names along which to partition the dataset.
    :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
    :param            scale: The column spans as generated before.
    :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
    :returns               : A list of valid partitions that cover the entire dataframe.
    """
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column, k) or not is_valid(df, rp, sensitive_column, k):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

def agg_categorical_column(series):
    return [','.join(set(series))]

def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        aggregations[column] = agg_categorical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)

def drop_grouped_rows(kanon_df):

    # Separate X and y
    kanon_X, kanon_y = kanon_df.drop(["label", "count"], axis=1), kanon_df["label"]

    # Helper function to remove grouped rows
    def process_partitions(row, i):
        if len(row[i][0]) != 1:
            return np.nan
        else:
            return int(row[i][0])

    # Apply helper function
    kanon_df = kanon_X.copy()
    for i, col in enumerate(kanon_X.columns):
        kanon_df[col] = kanon_X.apply(lambda row: process_partitions(row, i), axis=1)

    # Add the label back and drop null values
    kanon_df["label"] = kanon_y
    kanon_df = kanon_df.dropna()

    # Convert all columns to integers
    for col in kanon_df.columns:
        kanon_df[col] = kanon_df[col].apply(int)

    return kanon_df