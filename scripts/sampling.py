import numpy as np
import pandas as pd


class SampleProxyVoters:
    """Samples candidates from election data to use as proxy voters."""

    def __init__(self, fraction, sample_n, random_state=42):
        self.fraction = fraction
        self.random_state = random_state
        self.sample_n = sample_n

    def sample_proxy_voters_relative(
        self, fv_data, fv_results, fraction=None, random_state=None
    ):
        """Sample a fraction of candidates weighted by their party's vote share.

        Args:
            fv_data: long-format candidate DataFrame
            fv_results: election results DataFrame with 'Partinavn' and 'Stemmetal' columns
            fraction: fraction of candidates to sample; falls back to self.fraction
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        fraction = self.fraction if fraction is None else fraction
        random_state = self.random_state if random_state is None else random_state

        vote_shares = (
            fv_results.groupby("Partinavn")[["Stemmetal"]]
            .sum()
            .assign(Normalized=lambda x: x["Stemmetal"] / x["Stemmetal"].sum())
        )

        vote_shares = vote_shares.rename(
            index={
                "Enhedslisten - De Rød-Grønne": "Enhedslisten – De Rød-Grønne",
                "Danmarksdemokraterne - Inger Støjberg": "Danmarksdemokraterne ‒ Inger Støjberg",
                "Venstre": "Venstre, Danmarks Liberale Parti",
                "(Uden for parti)": "Uden for parti",
            }
        )

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()
        weights = candidates["Parti"].map(vote_shares["Normalized"])

        sampled_candidates = candidates.sample(
            frac=fraction,
            weights=weights,
            random_state=random_state,
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters

    # consider removing "uden for partierne" for larger sample size
    def sample_proxy_voters_equal_per_party(
        self, fv_data, fraction=None, random_state=None
    ):
        """Sample an equal number of candidates from each party, capped by the smallest party.

        Args:
            fv_data: long-format candidate DataFrame
            fraction: target fraction of total candidates; determines n_per_party
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        fraction = self.fraction if fraction is None else fraction
        random_state = self.random_state if random_state is None else random_state

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()

        n_total = int(len(candidates) * fraction)
        n_parties = candidates["Parti"].nunique()
        n_per_party = max(1, n_total // n_parties)

        min_party_size = candidates.groupby("Parti").size().min()
        n_per_party = min(n_per_party, min_party_size)

        sampled_candidates = (
            candidates.groupby("Parti")
            .apply(lambda g: g.sample(n=n_per_party, random_state=random_state))
            .reset_index(drop=True)
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters

    def sample_proxy_voters_n_per_party(
        self, fv_data, sample_n=None, random_state=None
    ):
        """Sample exactly n candidates per party with replacement.

        Args:
            fv_data: long-format candidate DataFrame
            sample_n: number of candidates to sample per party; falls back to self.sample_n
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        sample_n = self.sample_n if sample_n is None else sample_n
        random_state = self.random_state if random_state is None else random_state

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()

        sampled_candidates = (
            candidates.groupby("Parti")
            .apply(
                lambda g: g.sample(n=sample_n, replace=True, random_state=random_state),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters

    def sample_proxy_voters_random(self, fv_data, fraction=None, random_state=None):
        """Sample a random fraction of candidates with no weighting.

        Args:
            fv_data: long-format candidate DataFrame
            fraction: fraction of candidates to sample; falls back to self.fraction
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        fraction = self.fraction if fraction is None else fraction
        random_state = self.random_state if random_state is None else random_state

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()

        sampled_candidates = candidates.sample(
            frac=fraction,
            random_state=random_state,
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters

    def generate_random_proxy_voters(
        self, fv_data, sample_n=None, random_state=None, alpha=None
    ):
        """Generate synthetic proxy voters with uniformly random answers.

        Args:
            fv_data: long-format candidate DataFrame (used for question list)
            sample_n: number of synthetic voters to generate; falls back to self.sample_n
            random_state: random seed; falls back to self.random_state
            alpha: scale parameter determining the extreme answer values (±(1 + alpha))

        Returns:
            Long-format DataFrame of synthetic voters with a 'sample_id' column.
        """
        sample_n = self.sample_n if sample_n is None else sample_n
        random_state = self.random_state if random_state is None else random_state
        random_generator = np.random.default_rng(random_state)

        question_id_map = (
            fv_data[["Spørgsmål", "Spørgsmål_ID"]]
            .drop_duplicates()
            .set_index("Spørgsmål")["Spørgsmål_ID"]
        )
        questions = fv_data["Spørgsmål"].unique()
        answer_values = [-1 - alpha, -1, 1, 1 + alpha]

        rows = []
        for i in range(sample_n):
            name = f"Random_Voter_{i}"
            for question in questions:
                rows.append({
                    "Navn": name,
                    "Parti": "Random",
                    "sample_id": i + 1,
                    "Spørgsmål": question,
                    "Spørgsmål_ID": question_id_map[question],
                    "Svar_mapped": random_generator.choice(answer_values),
                })

        return pd.DataFrame(rows)

    def sample_proxy_voters_relative_number_candidates(
        self, fv_data, fraction=None, random_state=None
    ):
        """Sample a fraction of candidates weighted by each party's share of total candidates.

        Args:
            fv_data: long-format candidate DataFrame
            fraction: fraction of candidates to sample; falls back to self.fraction
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        fraction = self.fraction if fraction is None else fraction
        random_state = self.random_state if random_state is None else random_state

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()

        # Count number of candidates per party
        party_counts = candidates.groupby("Parti").size()
        # Calculate weights
        weights = party_counts[candidates["Parti"]].values / party_counts.sum()

        sampled_candidates = candidates.sample(
            frac=fraction,
            weights=weights,
            random_state=random_state,
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters

    def sample_proxy_voters_relative_n_total(
        self, fv_data, fv_results, sample_n=None, random_state=None
    ):
        """Sample a fixed total of candidates allocated per party by vote share, with replacement.

        Args:
            fv_data: long-format candidate DataFrame
            fv_results: election results DataFrame with 'Partinavn' and 'Stemmetal' columns
            sample_n: total number of candidates to sample; falls back to self.sample_n
            random_state: random seed; falls back to self.random_state

        Returns:
            Long-format DataFrame of sampled proxy voters with a 'sample_id' column.
        """
        sample_n = self.sample_n if sample_n is None else sample_n
        random_state = self.random_state if random_state is None else random_state

        vote_shares = (
            fv_results.groupby("Partinavn")[["Stemmetal"]]
            .sum()
            .assign(Normalized=lambda x: x["Stemmetal"] / x["Stemmetal"].sum())
        )

        vote_shares = vote_shares.rename(
            index={
                "Enhedslisten - De Rød-Grønne": "Enhedslisten – De Rød-Grønne",
                "Danmarksdemokraterne - Inger Støjberg": "Danmarksdemokraterne ‒ Inger Støjberg",
                "Venstre": "Venstre, Danmarks Liberale Parti",
                "(Uden for parti)": "Uden for parti",
            }
        )

        # Calculate exact number of samples per party
        party_sample_counts = (vote_shares["Normalized"] * sample_n).round().astype(int)

        # Rounding may cause total to drift from sample_n — fix by adjusting largest party
        diff = sample_n - party_sample_counts.sum()
        if diff != 0:
            largest_party = party_sample_counts.idxmax()
            party_sample_counts[largest_party] += diff

        candidates = fv_data[["Navn", "Parti"]].drop_duplicates()

        # Sample exact n per party with replacement
        sampled_candidates = (
            candidates.groupby("Parti")
            .apply(
                lambda g: g.sample(
                    n=party_sample_counts.get(g.name, 0),
                    replace=True,
                    random_state=random_state,
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

        rows = []
        for idx, row in sampled_candidates.iterrows():
            candidate_rows = fv_data[fv_data["Navn"] == row["Navn"]].copy()
            candidate_rows["sample_id"] = idx + 1
            rows.append(candidate_rows)

        proxy_voters = pd.concat(rows, ignore_index=True)

        return proxy_voters
