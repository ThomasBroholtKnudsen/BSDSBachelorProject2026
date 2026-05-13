import numpy as np

PARTY_LABEL_MAP = {
    "Venstre, Danmarks Liberale Parti": "Venstre",
    "SF - Socialistisk Folkeparti": "SF",
    "Enhedslisten – De Rød-Grønne": "Enhedslisten",
    "Danmarksdemokraterne ‒ Inger Støjberg": "Danmarksdemokraterne",
    "Det Konservative Folkeparti": "Konservative",
    "Borgernes Parti - Lars Boje Mathiesen": "Borgernes Parti",
}


def tied_top_parties(row, match_columns):
    """Return all parties tied at Match 1's relative agreement, consistent with single_alpha_match_counts."""
    first_distance = row["Match 1"][3]
    return [
        row[col][1]
        for col in match_columns
        if row[col] is not None and np.isclose(row[col][3], first_distance)
    ]

def tied_top_candidates(row, match_columns):
    """Return all candidates tied at Match 1's relative agreement, consistent with single_alpha_match_counts."""
    first_distance = row["Match 1"][3]
    return [
        row[col][0]
        for col in match_columns
        if row[col] is not None and np.isclose(row[col][3], first_distance)
    ]

def match_changes_baseline_computed(
    baseline_voters_with_distance,
    voters_with_distance,
):
    """Compare baseline and current top matches to measure how often party/candidate matches change.
    Uses tied_top_parties for both baseline and current sets for fair comparison.
    Only used for switch frequency calculations — voter travel uses match_changes_sampled_baseline.

    Args:
        baseline_voters_with_distance: voters_with_distances DataFrame from the baseline run
        voters_with_distance: voters_with_distances DataFrame from the current run

    Returns:
        Tuple of (party_switch_frequency, candidate_switch_frequency).
    """
    party_switch_counter = 0
    candidate_switch_counter = 0

    total_voters = len(voters_with_distance)
    baseline_indexed = baseline_voters_with_distance.set_index("Vælger")
    match_columns = [
        col for col in voters_with_distance.columns if col.startswith("Match ")
    ]

    for _, row in voters_with_distance.iterrows():
        # Use tied_top_parties for BOTH baseline and current for fair comparison
        baseline_voter_party_set = set(
            tied_top_parties(baseline_indexed.loc[row["Vælger"]], match_columns)
        )
        current_parties_set = set(tied_top_parties(row, match_columns))

        if sorted(current_parties_set) != sorted(baseline_voter_party_set):
            party_switch_counter += 1

        baseline_voter_candidate_set = set(
            tied_top_candidates(baseline_indexed.loc[row["Vælger"]], match_columns)
        )
        current_candidate_set = set(tied_top_candidates(row, match_columns))

        if sorted(current_candidate_set) != sorted(baseline_voter_candidate_set):
            candidate_switch_counter += 1

    party_switch_frequency = party_switch_counter / total_voters
    candidate_switch_frequency = candidate_switch_counter / total_voters

    return party_switch_frequency, candidate_switch_frequency

def match_changes_baseline_sampled(
    proxy_voters_mapped_w_noise,
    voters_with_distance,
):
    party_switch_dict = {}
    baseline_party_counts = {}

    baseline_party_lookup = (
        proxy_voters_mapped_w_noise[["Navn", "sample_id", "noise_id", "Parti"]]
        .drop_duplicates(subset=["Navn", "sample_id", "noise_id"])
        .assign(Vælger=lambda x: x["Navn"] + "_" + x["sample_id"].astype(str) + "_" + x["noise_id"].astype(str))
        [["Vælger", "Parti"]]
        .set_index("Vælger")
    )
    match_columns = [
        col for col in voters_with_distance.columns if col.startswith("Match ")
    ]

    for _, row in voters_with_distance.iterrows():
        baseline_party = baseline_party_lookup.loc[row["Vælger"], "Parti"]
        baseline_party = PARTY_LABEL_MAP.get(baseline_party, baseline_party)

        # Count baseline voters per party
        baseline_party_counts[baseline_party] = baseline_party_counts.get(baseline_party, 0) + 1

        current_parties = tied_top_parties(row, match_columns)
        current_parties = [PARTY_LABEL_MAP.get(p, p) for p in current_parties]
        weight = 1 / len(current_parties)
        for current_party in current_parties:
            party_switch_dict.setdefault(baseline_party, []).append((
                current_party,
                weight,
            ))

    return party_switch_dict, baseline_party_counts
