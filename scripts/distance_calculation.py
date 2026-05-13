from utils import shorten_party_labels

import pandas as pd
import numpy as np


class DistanceAnalysis:
    """Computes candidate-voter distances and match counts across alpha and noise values."""

    def __init__(
        self,
        candidate_party_table,
        n_matches,
        noise_iterations,
        random_state,
    ):
        self.candidate_party_table = candidate_party_table
        self.n_matches = n_matches
        self.noise_iterations = noise_iterations
        self.random_state = random_state

    def convert_to_wide_format(self, mapped_data, type):
        """Pivot mapped response data to wide format.

        Args:
            mapped_data: long-format DataFrame with mapped responses
            type: 'voter' or 'candidate'

        Returns:
            Wide-format DataFrame with individuals as rows and questions as columns.
        """

        if type == "voter":
            voter_data_wide = mapped_data.pivot(
                index=["Navn", "sample_id", "noise_id"],
                columns="Spørgsmål_ID",
                values="Svar_mapped",
            )
            return voter_data_wide
        elif type == "candidate":
            candidate_data_wide = mapped_data.pivot(
                index="Navn", columns="Spørgsmål_ID", values="Svar_mapped"
            )
            return candidate_data_wide
        else:
            raise ValueError("type must be 'voter' or 'candidate'")

    def single_voter_distance_calculation(
        self, voter_data_wide, candidate_data_wide, voter, sample_id, alpha
    ):
        """Compute distances from one voter to all candidates.

        Args:
            voter_data_wide: wide-format voter DataFrame
            candidate_data_wide: wide-format candidate DataFrame
            voter: voter name string
            sample_id: sample identifier for the voter
            alpha: scale parameter for the maximum possible distance

        Returns:
            DataFrame with candidates, their party, cityblock distance, and relative agreement.
        """
        # set target as the voter with the given sample id
        target = voter_data_wide.xs((voter, sample_id), level=("Navn", "sample_id"))

        # drop rows in candidate data where name is target voter (in case voter is also a candidate)
        candidate_data = candidate_data_wide.drop(index=voter, errors="ignore")

        # drop target columns where value is NaN
        target = target.dropna(axis=1)

        # get list of columns in target
        target_columns = target.columns

        # filter candidate data to only include columns that are in target
        candidate_data = candidate_data[target_columns]

        # voter has no NaNs after dropna above, so ravel to 1-D array of shape (n_questions,)
        voter_arr = target.to_numpy(dtype=float).ravel()

        # candidate matrix of shape (n_candidates, n_questions); may contain NaNs
        cand_arr = candidate_data.to_numpy(dtype=float)

        # per-cell boolean mask: True where both voter and candidate answered the question.
        # voter_arr has no NaNs, so this reduces to ~np.isnan(cand_arr), shape (n_candidates, n_questions)
        valid_mask = ~np.isnan(cand_arr)

        # compute |voter - candidate| per cell; set NaN (unanswered) positions to 0 so they
        # contribute nothing to the sum — equivalent to cityblock over only valid columns
        diff = np.where(valid_mask, np.abs(cand_arr - voter_arr), 0.0)

        # sum absolute differences across questions to get cityblock distance per candidate, shape (n_candidates,)
        distances = diff.sum(axis=1)

        # count how many questions were compared per candidate, shape (n_candidates,)
        compared_questions = valid_mask.sum(axis=1)

        # maximum possible cityblock distance given alpha and number of compared questions
        max_distance = (2 * (alpha + 1)) * compared_questions

        # relative agreement in [0, 1]: 1 means identical answers, 0 means maximally different
        relative_agreement = (max_distance - distances) / max_distance

        # where no questions overlap, distance and relative agreement are undefined
        no_overlap = compared_questions == 0
        distances[no_overlap] = np.nan
        relative_agreement[no_overlap] = np.nan

        # map candidate names to party; reindex fills NaN for any name not in the table
        party_lookup = self.candidate_party_table.set_index("Navn")["Parti"]
        candidate_names = candidate_data.index.tolist()

        # build result dataframe directly from arrays — no row-by-row dict building needed
        distance_df = pd.DataFrame({
            "Kandidat": candidate_names,
            "Parti": party_lookup.reindex(candidate_names).values,
            "Distance til vælger": distances,
            "Relativ enighed": relative_agreement,
        })

        return distance_df

    def n_closest(self, distance_df):
        """Return the n closest candidates and any candidates tied at the cutoff distance.

        Args:
            distance_df: DataFrame with candidate distances from a single voter

        Returns:
            Tuple of (n_closest_df, tied_candidates_df).
        """

        # remove NaN values and sort by relative agreement descending (highest agreement first)
        sorted_distance_df = distance_df.dropna(subset=["Relativ enighed"]).sort_values(
            "Relativ enighed", ascending=False
        )

        if sorted_distance_df.empty:
            return sorted_distance_df, sorted_distance_df

        # handle requests larger than available candidates
        n = min(self.n_matches, len(sorted_distance_df))

        # get relative agreement at the n-th position (cutoff)
        cutoff_distance = sorted_distance_df.iloc[n - 1]["Relativ enighed"]

        # get candidates with strictly higher relative agreement than the cutoff
        strictly_closer_cand = sorted_distance_df[
            sorted_distance_df["Relativ enighed"] > cutoff_distance
        ]

        # get candidates tied at the cutoff relative agreement
        tied_cand = sorted_distance_df[
            sorted_distance_df["Relativ enighed"] == cutoff_distance
        ]

        # number of remaining spots to fill from the tied group
        remaining_cand = n - len(strictly_closer_cand)

        # if the tied candidates exactly fill the remaining spots, take all of them without random sampling
        if remaining_cand == len(tied_cand):
            n_closest_cand = pd.concat([strictly_closer_cand, tied_cand], axis=0)
            tied_cand = tied_cand.drop(tied_cand.index[: len(tied_cand)])
        else:
            # randomly sample from the tied group to fill remaining spots
            sampled_ties = tied_cand.sample(
                n=remaining_cand, random_state=self.random_state
            )

            # combine strictly better candidates with the sampled tied candidates
            n_closest_cand = pd.concat([strictly_closer_cand, sampled_ties], axis=0)

        # sort by relative agreement descending, randomize order within tied groups
        np.random.seed(self.random_state)
        n_closest_cand = (
            n_closest_cand.assign(_tie_random=np.random.random(len(n_closest_cand)))
            .sort_values(["Relativ enighed", "_tie_random"], ascending=[False, False])
            .drop(columns="_tie_random")
            .reset_index(drop=True)
        )

        return n_closest_cand, tied_cand

    def all_voters_distance_calculation(
        self, voter_data_wide, candidate_data_wide, alpha
    ):
        """Compute top-n matches for every voter across all noise iterations.

        Args:
            voter_data_wide: wide-format voter DataFrame
            candidate_data_wide: wide-format candidate DataFrame
            alpha: scale parameter for the maximum possible distance

        Returns:
            Tuple of (voters_with_distances DataFrame, dict of per-voter distance DataFrames).
        """

        # create dataframe with voters names, their top n matches and the corresponding distances
        rows = []

        # dict keyed by "voter_sampleid_noiseid" with the full distance DataFrame per voter
        distance_dfs = {}
        for i in (
            range(self.noise_iterations)
            if self.noise_iterations is not None
            else range(1)
        ):
            voter_data_single_iteration = voter_data_wide[
                voter_data_wide.index.get_level_values("noise_id") == (i + 1)
            ]

            for row in voter_data_single_iteration.itertuples():
                voter = row.Index[0]
                sample_id = row.Index[1]
                noise_id = row.Index[2]
                distance_df = self.single_voter_distance_calculation(
                    voter_data_single_iteration,
                    candidate_data_wide,
                    voter,
                    sample_id,
                    alpha,
                )

                distance_dfs[f"{voter}_{sample_id}_{noise_id}"] = (
                    distance_df.sort_values("Relativ enighed", ascending=False)
                )

                top_n_closest, tied_cand = self.n_closest(distance_df)

                match_tuples = list(top_n_closest.itertuples(index=False, name=None))

                row = {"Vælger": f"{voter}_{sample_id}_{noise_id}"}
                for j in range(self.n_matches):
                    row[f"Match {j + 1}"] = (
                        match_tuples[j] if j < len(match_tuples) else None
                    )

                row["Tied kandidater"] = list(tied_cand["Kandidat"])
                rows.append(row)

        voters_with_distances = pd.DataFrame(rows)

        return voters_with_distances, distance_dfs

    def single_alpha_match_counts(self, voters_with_distances):
        """Count best-match hits per candidate and party for a single alpha/noise setting.

        Args:
            voters_with_distances: DataFrame of voters with their top-n match columns

        Returns:
            Tuple of (candidate_match_counts DataFrame, party_match_counts DataFrame).
        """

        all_parties = [
            "Socialdemokratiet",
            "Venstre, Danmarks Liberale Parti",
            "SF - Socialistisk Folkeparti",
            "Enhedslisten – De Rød-Grønne",
            "Radikale Venstre",
            "Dansk Folkeparti",
            "Alternativet",
            "Danmarksdemokraterne ‒ Inger Støjberg",
            "Det Konservative Folkeparti",
            "Liberal Alliance",
            "Moderaterne",
            "Borgernes Parti - Lars Boje Mathiesen",
            "Uden for parti",
        ]

        # count candidates that are Match 1, including subsequent matches tied on distance with Match 1
        match_columns = [
            f"Match {i}"
            for i in range(1, self.n_matches + 1)
            if f"Match {i}" in voters_with_distances.columns
        ]

        counted_matches = []

        for _, voter_row in voters_with_distances[match_columns].iterrows():
            first_match = voter_row.get("Match 1")

            first_distance = first_match[3]

            for column_name in match_columns:
                current_match = voter_row[column_name]

                if not np.isclose(current_match[3], first_distance):
                    break

                counted_matches.append((current_match[0], current_match[1]))

        candidate_match_counts = (
            pd.DataFrame(counted_matches, columns=["Kandidat", "Parti"])
            .value_counts(["Kandidat", "Parti"])
            .reset_index(name="Count")
        )

        # create dataframe with counts per party from candidate match counts
        party_match_counts = (
            candidate_match_counts.groupby("Parti")["Count"].sum().reset_index()
        )

        # Fill in parties with zero matches to ensure they are included in the plot
        party_match_counts = (
            party_match_counts.set_index("Parti")
            .reindex(all_parties, fill_value=0)
            .reset_index()
        )

        # make party match counts relative to total number of counts
        party_match_counts["Relative Count"] = (
            party_match_counts["Count"] / party_match_counts["Count"].sum()
        )

        # make candidate match counts relative to total number of counts
        candidate_match_counts["Relative Count"] = (
            candidate_match_counts["Count"] / candidate_match_counts["Count"].sum()
        )

        # order parties by relative count
        party_match_counts = party_match_counts.sort_values(
            "Relative Count", ascending=False
        )

        return candidate_match_counts, party_match_counts

    def multiple_alpha_match_counts(
        self,
        match_counts_dict,
        alpha_values,
        noise_values,
        simulations=None,
    ):
        """Aggregate match counts across multiple alpha and noise values.

        Args:
            match_counts_dict: dict mapping alpha/noise keys to (candidate_counts, party_counts) tuples
            alpha_values: list of alpha values to aggregate over
            noise_values: list of noise levels to aggregate over
            simulations: number of simulations

        Returns:
            For 'distance_multiple_sim': party_match_counts DataFrame with a simulation column.
        """

        party_dfs = []
        candidate_dfs = []

        for simulation in range(simulations):
            for alpha in alpha_values:
                for noise in noise_values:
                    key = f"alpha:{alpha}_noise:{noise}_simulation-number:{simulation + 1}"
                    candidate_match_counts = match_counts_dict[key][0]
                    party_match_counts = match_counts_dict[key][1]

                    party_dfs.append(
                        party_match_counts.assign(
                            Alpha=alpha, Noise=noise, Simulation=simulation + 1
                        )
                    )
                    candidate_dfs.append(
                        candidate_match_counts.assign(
                            Alpha=alpha, Noise=noise, Simulation=simulation + 1
                        )
                    )

        alpha_noise_simulation_party_match_counts = pd.concat(
            party_dfs, ignore_index=True
        )
        alpha_noise_simulation_candidate_match_counts = pd.concat(
            candidate_dfs, ignore_index=True
        )

        alpha_noise_simulation_party_match_counts = shorten_party_labels(
            alpha_noise_simulation_party_match_counts
        )
        alpha_noise_simulation_candidate_match_counts = shorten_party_labels(
            alpha_noise_simulation_candidate_match_counts
        )

        alpha_noise_simulation_party_match_counts = (
            alpha_noise_simulation_party_match_counts.sort_values([
                "Parti",
                "Alpha",
                "Noise",
            ])
        )
        alpha_noise_simulation_candidate_match_counts = (
            alpha_noise_simulation_candidate_match_counts.sort_values([
                "Kandidat",
                "Alpha",
                "Noise",
            ])
        )

        return (
            alpha_noise_simulation_party_match_counts,
            alpha_noise_simulation_candidate_match_counts,
        )
