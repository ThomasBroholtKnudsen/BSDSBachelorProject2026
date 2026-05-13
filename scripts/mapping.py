def set_distance(alpha):
    """Build a response-to-numeric mapping for the given alpha and election type.

    Args:
        alpha: non-negative int or float; controls the weight of extreme responses

    Returns:
        Dict mapping Danish response strings to numeric values.
    """

    # Check input
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("alpha must be a non-negative number or math.inf")

    # set distances for FV
    # create mapping dictionary with default values for moderate response options
    mapping_FV = {
        "Uenig": -1,
        "Enig": 1,
    }

    # convert moderate responses to floats if alpha is a float
    if type(alpha) is float:
        mapping_FV["Uenig"] = float(mapping_FV["Uenig"])
        mapping_FV["Enig"] = float(mapping_FV["Enig"])

    # add extreme response options to mapping dictionary
    mapping_FV["Helt uenig"] = mapping_FV["Uenig"] - alpha
    mapping_FV["Helt enig"] = mapping_FV["Enig"] + alpha

    return mapping_FV


def map_responses(data, mapping):
    """Apply a response mapping to the 'Svar' column and return a new DataFrame.

    Args:
        data: DataFrame with a 'Svar' column of raw response strings
        mapping: dict from response strings to numeric values (from set_distance)

    Returns:
        Copy of data with an added 'Svar_mapped' column.
    """

    # Return a new dataframe so repeated alpha mappings don't overwrite each other.
    mapped_data = data.copy()
    mapped_data["Svar_mapped"] = mapped_data["Svar"].map(mapping)
    return mapped_data
