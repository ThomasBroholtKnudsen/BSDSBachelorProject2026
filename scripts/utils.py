import pandas as pd
from scipy import stats

def load_data(dataset, file_type="excel"):
    """Load a dataset from an Excel or CSV file.

    Args:
        dataset: file path to the dataset
        file_type: 'excel' or 'csv'

    Returns:
        DataFrame with the loaded data.
    """

    if file_type == "excel":
        data = pd.read_excel(dataset)
    elif file_type == "csv":
        data = pd.read_csv(dataset)
    else:
        raise ValueError("file_type must be 'excel' or 'csv'")
    return data

def question_id(dataset):
    """Add a numeric 'Spørgsmål_ID' column mapping each unique question to an integer.

    Args:
        dataset: DataFrame with a 'Spørgsmål' column

    Returns:
        The same DataFrame with an added 'Spørgsmål_ID' column.
    """
    unique_questions = dataset["Spørgsmål"].unique()

    question_id_dict = {question: id for id, question in enumerate(unique_questions)}

    dataset["Spørgsmål_ID"] = dataset["Spørgsmål"].map(question_id_dict)

    return dataset

def candidate_party_table(dataset):
    """Build a table with one row per candidate and their party.

    Args:
        dataset: raw long-format DataFrame with 'Navn' and 'Parti' columns

    Returns:
        DataFrame with one row per candidate and their party.
    """

    candidate_party = dataset.groupby("Navn")["Parti"].first().reset_index()

    return candidate_party


PARTY_LABEL_MAP = {
    "Venstre, Danmarks Liberale Parti": "Venstre",
    "SF - Socialistisk Folkeparti": "SF",
    "Enhedslisten – De Rød-Grønne": "Enhedslisten",
    "Danmarksdemokraterne ‒ Inger Støjberg": "Danmarksdemokraterne",
    "Det Konservative Folkeparti": "Konservative",
    "Borgernes Parti - Lars Boje Mathiesen": "Borgernes Parti",
}

def shorten_party_labels(df, col="Parti"):
    """Replace full party names in a DataFrame column with shortened labels.

    Args:
        df: DataFrame containing the party name column
        col: name of the column to apply the mapping to

    Returns:
        Copy of df with shortened party names in the specified column.
    """
    PARTY_LABEL_MAP = {
    "Venstre, Danmarks Liberale Parti": "Venstre",
    "SF - Socialistisk Folkeparti": "SF",
    "Enhedslisten – De Rød-Grønne": "Enhedslisten",
    "Danmarksdemokraterne ‒ Inger Støjberg": "Danmarksdemokraterne",
    "Det Konservative Folkeparti": "Konservative",
    "Borgernes Parti - Lars Boje Mathiesen": "Borgernes Parti",
    }
    df = df.copy()
    df[col] = df[col].replace(PARTY_LABEL_MAP)  # only mapped labels change
    return df

def compute_mean_ci_se(df, group_cols, value_col="Relative Count"):
    """Compute mean, SE, and 95% CI across simulations for a grouped DataFrame.

    Args:
        df: DataFrame with simulation-level data
        group_cols: list of columns to group by, e.g. ["Parti", "Alpha", "Noise"]
        value_col: column to aggregate

    Returns:
        DataFrame with group_cols plus Mean, SE, CI, Lower_CI, Upper_CI, Lower_SE, Upper_SE, N columns.
    """
    summary = []
    for group_vals, group_df in df.groupby(group_cols):
        values = group_df[value_col].values
        n = len(values)
        mean = values.mean()
        se = 0
        ci = 0
        if n > 1:
            se = stats.sem(values)
            ci = se * stats.t.ppf(0.975, df=n - 1)

        # Handle single or multiple group columns
        if isinstance(group_vals, tuple):
            row = dict(zip(group_cols, group_vals))
        else:
            row = {group_cols[0]: group_vals}

        row.update({
            "Mean": mean,
            "SE": se if n > 1 else 0,
            "CI": ci,
            "Lower_CI": max(0, mean - ci),
            "Upper_CI": mean + ci,
            "Lower_SE": max(0, mean - (se if n > 1 else 0)),
            "Upper_SE": mean + (se if n > 1 else 0),
            "N": n,
        })
        summary.append(row)

    return pd.DataFrame(summary)