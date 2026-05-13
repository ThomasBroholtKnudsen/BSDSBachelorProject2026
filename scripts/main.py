# import functions from scripts
from sampling import SampleProxyVoters
from noise import NoiseToProxyVoters
from utils import load_data, question_id, candidate_party_table, compute_mean_ci_se
from mapping import set_distance, map_responses
from distance_calculation import DistanceAnalysis
from plots import Plotting
from match_change import (
    match_changes_baseline_computed,
    match_changes_baseline_sampled,
)

import os
import numpy as np
import pandas as pd

from pca import PCAAnalysis


class MasterFunction:
    """Orchestrates the full analysis pipeline: data loading, sampling, noise, distance calculation, and plotting."""

    def __init__(
        self,
        dataset="../data/raw/FV_26_Thomas_Christian.xlsx",
        alpha_values=[
            0.25,
            0.5,
            0.75,
            1,
            1.5,
            2.0,
            3.0,
        ],
        election_results_path="../data/interim/fv_results_26.csv",
        sample_fraction=None,
        sample_n=2500,
        sample_type="n_total_relative_to_votes",
        noise_values=[0.0, 0.2, 0.3, 0.4],
        noise_iterations=None,
        noise_rounds_per_question=4,
        n_matches=10,
        random_state=42,
        baseline_alpha=1,
        simulations=1,
    ):
        self.dataset = dataset
        self.alpha_values = alpha_values
        self.election_results_path = election_results_path
        self.sample_fraction = sample_fraction
        self.sample_n = sample_n
        self.sample_type = sample_type
        self.noise_values = noise_values
        self.noise_iterations = noise_iterations
        self.noise_rounds_per_question = noise_rounds_per_question
        self.n_matches = n_matches
        self.random_state = random_state
        self.experiment_folder = None
        self.baseline_alpha = baseline_alpha
        self.simulations = simulations

    def register_experiment(self):
        """Create the experiment output folder structure and write a parameters.txt file."""
        # create experiment folder with distinct name based on parameters
        print(
            f"Registering experiment: Simulations={self.simulations} | alphas={self.alpha_values} | noise={self.noise_values}"
        )
        self.experiment_folder = f"../experiments/simulations_{self.simulations}_alpha_{min(self.alpha_values)}-{max(self.alpha_values)}_noise_{min(self.noise_values)}-{max(self.noise_values)}"
        os.makedirs(self.experiment_folder, exist_ok=True)

        os.makedirs(os.path.join(self.experiment_folder, "tables"), exist_ok=True)

        os.makedirs(
            os.path.join(self.experiment_folder, "tables/voters_with_distances"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder,
                "tables/party_relative_match_count_relative_change",
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "tables/party_match_count_deviations"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "tables/voter_retention"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "party_distribution_plots"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "voter_travel_plots"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder,
                "voter_travel_plots/voter_travel_sankey_diagrams",
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder, "voter_travel_plots/voter_travel_heatmaps"
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "switch_frequency_plots"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "all_parties_alpha_comparison"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder, "tables/proxy_voters_mapped_with_noise"
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "candidate_match_bar_plots"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(self.experiment_folder, "per_party_alpha_comparison"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder,
                "per_party_alpha_comparison/relative_match_counts",
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder, "per_party_alpha_comparison/voter_retention"
            ),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(
                self.experiment_folder,
                "per_party_alpha_comparison/relative_match_count_relative_change_from_baseline",
            ),
            exist_ok=True,
        )

        if self.simulations == 1:
            os.makedirs(
                os.path.join(self.experiment_folder, "pca_candidate_plots"),
                exist_ok=True,
            )

            os.makedirs(
                os.path.join(self.experiment_folder, "pca_voter_plots"), exist_ok=True
            )

        # create file with list of parameters used in the experiment
        with open(os.path.join(self.experiment_folder, "parameters.txt"), "w") as f:
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Alpha values: {self.alpha_values}\n")
            f.write(f"Election results path: {self.election_results_path}\n")
            f.write(f"Sample fraction: {self.sample_fraction}\n")
            f.write(f"Sample n: {self.sample_n}\n")
            f.write(f"Sample type: {self.sample_type}\n")
            f.write(f"Noise levels: {self.noise_values}\n")
            f.write(f"Noise iterations: {self.noise_iterations}\n")
            f.write(f"Noise rounds per question: {self.noise_rounds_per_question}\n")
            f.write(f"Number of matches: {self.n_matches}\n")
            f.write(f"Random state: {self.random_state}\n")
            f.write(f"Baseline alpha: {self.baseline_alpha}\n")
            f.write(f"Simulations: {self.simulations}\n")

        return None

    def prepare_data(self):
        """Load and clean the raw dataset, add question IDs, and build supporting lookup tables.

        Returns:
            Tuple of (data_q_id, election_results, candidates_votes, candidate_party, candidates_per_party).
        """
        print("Loading data...")
        data = load_data(self.dataset, file_type="excel")
        print("Filtering candidates to those with exactly 29 questions...")
        question_counts = data.groupby("Navn")["Spørgsmål"].nunique()
        valid_candidates = question_counts[question_counts == 29].index
        data = data[data["Navn"].isin(valid_candidates)]

        candidates_per_party = (
            data.groupby("Parti")["Navn"].nunique().sort_values(ascending=False)
        )

        # handle candidates who share a name but belong to different parties
        duplicate_names = data.groupby("Navn")["Parti"].nunique()
        dup_names = duplicate_names[duplicate_names > 1].index
        if len(dup_names) > 0:
            print(
                f"Handle {len(dup_names)} candidate name(s) shared across parties: {list(dup_names)}"
            )
            mask = data["Navn"].isin(dup_names)
            data.loc[mask, "Navn"] = (
                data.loc[mask, "Navn"] + " (" + data.loc[mask, "Parti"] + ")"
            )

        print("Loading election results...")
        election_results = load_data(self.election_results_path, file_type="csv")

        candidates_votes = (
            election_results.groupby("Navn", as_index=False)["Stemmetal"]
            .sum()
            .assign(Stemmetal_log=lambda d: np.log(d["Stemmetal"]))
            .set_index("Navn")
        )

        print("Creating candidate-party table...")
        candidate_party = candidate_party_table(data)

        print("Adding question IDs...")
        data_q_id = question_id(data)

        return (
            data_q_id,
            election_results,
            candidates_votes,
            candidate_party,
            candidates_per_party,
        )

    def map_data(self, data_q_id):
        """Apply each alpha's response mapping to the dataset.

        Returns:
            Dict mapping each alpha value to its mapped DataFrame.
        """
        print("Mapping data for all alpha values...")
        data_mapped_dict = {}
        for alpha in self.alpha_values:
            mapping = set_distance(alpha)
            data_mapped = map_responses(data_q_id, mapping)
            data_mapped_dict[alpha] = data_mapped

        return data_mapped_dict

    def sample_proxy_voters(
        self, data_mapped, election_results, alpha, random_state=None
    ):
        """Sample proxy voters from the mapped candidate data using the configured sample_type.

        Returns:
            Long-format DataFrame of sampled proxy voters with mapped responses.
        """
        random_state = random_state if random_state is not None else self.random_state
        sampler = SampleProxyVoters(
            fraction=self.sample_fraction,
            sample_n=self.sample_n,
            random_state=random_state,
        )

        if self.sample_type == "relative":
            print("Sampling proxy voters relative to vote shares...")
            proxy_voters_mapped = sampler.sample_proxy_voters_relative(
                data_mapped, election_results
            )
        elif self.sample_type == "equal_per_party":
            print("Sampling proxy voters equal per party...")
            proxy_voters_mapped = sampler.sample_proxy_voters_equal_per_party(
                data_mapped
            )
        elif self.sample_type == "random":
            print("Sampling proxy voters randomly...")
            proxy_voters_mapped = sampler.sample_proxy_voters_random(data_mapped)

        elif self.sample_type == "relative_to_number_candidates":
            print("Sampling proxy voters relative to number of candidates...")
            proxy_voters_mapped = (
                sampler.sample_proxy_voters_relative_number_candidates(data_mapped)
            )

        elif self.sample_type == "random_generated":
            print("Generating random proxy voters...")
            proxy_voters_mapped = sampler.generate_random_proxy_voters(
                data_mapped, alpha=alpha
            )

        elif self.sample_type == "n_per_party":
            print("Sampling proxy voters n per party...")
            proxy_voters_mapped = sampler.sample_proxy_voters_n_per_party(data_mapped)

        elif self.sample_type == "n_total_relative_to_votes":
            print("Sampling proxy voters n total relative to votes...")
            proxy_voters_mapped = sampler.sample_proxy_voters_relative_n_total(
                data_mapped, election_results
            )

        return proxy_voters_mapped

    def add_noise(self, proxy_voters_mapped, alpha, noise, random_state=None):
        """Add noise to proxy voter responses using the configured noise method.

        Returns:
            Long-format DataFrame of proxy voters with a 'noise_id' column and perturbed responses.
        """
        random_state = random_state if random_state is not None else self.random_state
        noise_adder = NoiseToProxyVoters(noise, random_state=random_state)
        print(f"Adding noise to proxy voters with noise level {noise}...")
        if self.sample_type in [
            "relative",
            "equal_per_party",
            "random",
            "relative_to_number_candidates",
        ]:
            proxy_voters_mapped_w_noise, _ = noise_adder.iterative_noise_from_original(
                proxy_voters_mapped,
                self.noise_iterations,
                self.noise_rounds_per_question,
                alpha,
            )
        else:
            proxy_voters_mapped_w_noise, _ = noise_adder.add_noise(
                proxy_voters_mapped,
                noise,
                self.noise_rounds_per_question,
                alpha,
            )
            proxy_voters_mapped_w_noise["noise_id"] = 1

        return proxy_voters_mapped_w_noise

    def avg_relative_agreement_top_matches(self, voters_with_distances):
        """Calculate the average relative agreement of the top match (Match 1) across all voters."""
        agreements = []
        for _, row in voters_with_distances.iterrows():
            match = row.get("Match 1")
            if match is not None:
                agreement = match[3]
                if agreement is not None and not np.isnan(agreement):
                    agreements.append(agreement)
        return np.mean(agreements) if agreements else np.nan

    def analysis(
        self,
        data_mapped_dict,
        candidate_party_table,
        election_results,
        candidates_votes,
        candidates_per_party,
    ):
        """Run the full analysis pipeline and produce all output plots."""
        print("Performing distance analysis including confidence intervals...")
        # create a dictionary to store match_counts for each alpha, noise and simulation combination
        match_counts_per_alpha_noise_simulation_dict = {}
        switch_frequency_rows = []
        switch_matrix_rows = []
        avg_relative_agreement_rows = []

        for simulation in range(self.simulations):
            print(f"Running simulation {simulation + 1}/{self.simulations}...")
            sim_random_state = self.random_state + simulation * (
                self.noise_iterations + 1
            )

            distance_analysis = DistanceAnalysis(
                candidate_party_table=candidate_party_table,
                n_matches=self.n_matches,
                noise_iterations=self.noise_iterations,
                random_state=self.random_state,
            )

            computed_voters_with_distances = {}

            # run distance analysis for each alpha and noise combination and store results in dictionary
            for alpha in self.alpha_values:
                if self.simulations == 1:
                    print(f"Performing PCA analysis for alpha={alpha}...")
                    pca_analysis = PCAAnalysis(
                        data_mapped_dict[alpha],
                        self.experiment_folder,
                    )
                    (
                        pca_components,
                        parties,
                        explained_variance_ratio,
                        candidate_names,
                    ) = pca_analysis.perform_pca()
                    pca_analysis.pca_plots_candidates(
                        pca_components,
                        parties,
                        explained_variance_ratio,
                        alpha,
                        candidate_names=candidate_names,
                        highlight_name=None,
                    )
                    pca_analysis.pca_plots_candidates_3d(
                        pca_components,
                        parties,
                        explained_variance_ratio,
                        alpha,
                        candidate_names=candidate_names,
                        highlight_name=None,
                    )
                else:
                    pass

                candidate_data_wide = distance_analysis.convert_to_wide_format(
                    data_mapped_dict[alpha], "candidate"
                )

                for noise in self.noise_values:
                    print(f"alpha={alpha}, noise={noise}...")
                    proxy_voters_mapped = self.sample_proxy_voters(
                        data_mapped_dict[alpha],
                        election_results,
                        alpha,
                        random_state=sim_random_state,
                    )

                    proxy_voters_mapped_w_noise = self.add_noise(
                        proxy_voters_mapped,
                        alpha,
                        noise,
                        random_state=sim_random_state,
                    )

                    if self.simulations == 1:
                        pca_analysis.pca_plots_voters(
                            proxy_voters_mapped_w_noise,
                            alpha,
                            noise,
                            highlight_name=None,
                        )
                        pca_analysis.pca_plots_voters_3d(
                            proxy_voters_mapped_w_noise,
                            alpha,
                            noise,
                            highlight_name=None,
                        )
                    else:
                        pass

                    voter_data_wide = distance_analysis.convert_to_wide_format(
                        proxy_voters_mapped_w_noise, "voter"
                    )

                    voters_with_distances, distance_dfs = (
                        distance_analysis.all_voters_distance_calculation(
                            voter_data_wide, candidate_data_wide, alpha
                        )
                    )

                    avg_agreement = self.avg_relative_agreement_top_matches(
                        voters_with_distances
                    )
                    avg_relative_agreement_rows.append({
                        "Simulation": simulation + 1,
                        "Alpha": alpha,
                        "Noise": noise,
                        "Avg Relative Agreement (Top Match)": avg_agreement,
                    })

                    computed_voters_with_distances[(alpha, noise)] = {
                        "voters_with_distances": voters_with_distances,
                        "proxy_voters_mapped_w_noise": proxy_voters_mapped_w_noise,
                    }

                    # voters_with_distances.to_excel(
                    #     os.path.join(
                    #         self.experiment_folder,
                    #         f"tables/voters_with_distances/voters_with_distances_sim_{simulation + 1}_alpha_{alpha}_noise_{noise}.xlsx",
                    #     ),
                    #     index=False,
                    # )

                    candidate_match_counts, party_match_counts = (
                        distance_analysis.single_alpha_match_counts(
                            voters_with_distances
                        )
                    )

                    match_counts_per_alpha_noise_simulation_dict[
                        f"alpha:{alpha}_noise:{noise}_simulation-number:{simulation + 1}"
                    ] = (candidate_match_counts, party_match_counts)

            # After all alpha/noise combinations are computed for this simulation,
            # apply each baseline noise to get voter travel data and switch frequencies

            for noise in self.noise_values:
                baseline = computed_voters_with_distances[(self.baseline_alpha, noise)]

                for alpha in self.alpha_values:
                    current = computed_voters_with_distances[(alpha, noise)]

                    # Voter travel using sampled baseline
                    party_switch_dict_sampled, baseline_party_counts = (
                        match_changes_baseline_sampled(
                            baseline["proxy_voters_mapped_w_noise"],
                            current["voters_with_distances"],
                        )
                    )

                    for from_party, to_parties in party_switch_dict_sampled.items():
                        total_weight = sum(w for _, w in to_parties)
                        to_party_weights = {}
                        for to_party, weight in to_parties:
                            to_party_weights[to_party] = (
                                to_party_weights.get(to_party, 0) + weight
                            )
                        for to_party, weight in to_party_weights.items():
                            switch_matrix_rows.append({
                                "Alpha": alpha,
                                "Noise": noise,
                                "Simulation": simulation + 1,
                                "From": from_party,
                                "To": to_party,
                                "Relative": weight / total_weight
                                if total_weight > 0
                                else 0,
                                "Baseline Count": baseline_party_counts.get(
                                    from_party, 0
                                ),
                            })

            for baseline_noise in self.noise_values:
                baseline = computed_voters_with_distances[
                    (self.baseline_alpha, baseline_noise)
                ]

                for alpha in self.alpha_values:
                    current = computed_voters_with_distances[(alpha, baseline_noise)]

                    # Switch frequency using match-based baseline
                    party_switch_frequency, candidate_switch_frequency = (
                        match_changes_baseline_computed(
                            baseline["voters_with_distances"],
                            current["voters_with_distances"],
                        )
                    )
                    switch_frequency_rows.append({
                        "Alpha": alpha,
                        "Noise": baseline_noise,
                        "Baseline Alpha": self.baseline_alpha,
                        "Baseline Noise": baseline_noise,
                        "Simulation": simulation + 1,
                        "Party Switch Frequency": party_switch_frequency,
                        "Candidate Switch Frequency": candidate_switch_frequency,
                    })

                for noise in self.noise_values:
                    if noise == baseline_noise:
                        continue  # already computed in row loop
                    current = computed_voters_with_distances[
                        (self.baseline_alpha, noise)
                    ]
                    party_switch_frequency, candidate_switch_frequency = (
                        match_changes_baseline_computed(
                            baseline["voters_with_distances"],
                            current["voters_with_distances"],
                        )
                    )
                    switch_frequency_rows.append({
                        "Alpha": self.baseline_alpha,
                        "Noise": noise,
                        "Baseline Alpha": self.baseline_alpha,
                        "Baseline Noise": baseline_noise,
                        "Simulation": simulation + 1,
                        "Party Switch Frequency": party_switch_frequency,
                        "Candidate Switch Frequency": candidate_switch_frequency,
                    })

        print("All simulations complete. Generating output plots and statistics...")
        # combine match counts for all alpha values into two dataframes, one for candidates and one for parties
        (
            alpha_noise_simulation_party_match_counts,
            alpha_noise_simulation_candidate_match_counts,
        ) = distance_analysis.multiple_alpha_match_counts(
            match_counts_per_alpha_noise_simulation_dict,
            self.alpha_values,
            self.noise_values,
            simulations=self.simulations,
        )

        alpha_noise_simulation_party_match_counts.to_excel(
            os.path.join(
                self.experiment_folder,
                "tables/all_simulations_party_match_counts.xlsx",
            ),
            index=False,
        )

        party_match_summary = compute_mean_ci_se(
            df=alpha_noise_simulation_party_match_counts,
            group_cols=["Parti", "Alpha", "Noise"],
            value_col="Relative Count",
        )

        party_match_summary.to_excel(
            os.path.join(
                self.experiment_folder,
                "tables/party_match_summary.xlsx",
            )
        )

        # Create dataframe for switch matrix
        switch_matrix_df = pd.DataFrame(switch_matrix_rows)

        all_to = switch_matrix_df["To"].unique()
        all_simulations = switch_matrix_df["Simulation"].unique()
        all_alphas = switch_matrix_df["Alpha"].unique()
        all_noises = switch_matrix_df["Noise"].unique()

        n_simulations = switch_matrix_df["Simulation"].nunique()

        # filter switch_matrix_df to only include from parties that have baseline voters in all simulations for each alpha and noise combination to ensure we are comparing the same set of from parties across simulations when computing confidence intervals for the switch matrix
        valid_from_parties = (
            switch_matrix_df[switch_matrix_df["Baseline Count"] > 0]
            .groupby(["Noise", "From"])["Simulation"]
            .nunique()
            .reset_index()
            .rename(columns={"Simulation": "N Simulations With Baseline Voters"})
        )

        valid_from_parties = valid_from_parties[
            valid_from_parties["N Simulations With Baseline Voters"] == n_simulations
        ]["From"].unique()

        switch_matrix_df = switch_matrix_df[
            switch_matrix_df["From"].isin(valid_from_parties)
        ]

        full_index = pd.MultiIndex.from_product(
            [all_alphas, all_noises, all_simulations, valid_from_parties, all_to],
            names=["Alpha", "Noise", "Simulation", "From", "To"],
        )

        switch_matrix_df = (
            switch_matrix_df.set_index([
                "Alpha",
                "Noise",
                "Simulation",
                "From",
                "To",
            ])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        # Create summary dataframe for switch matrix and compute mean, confidence intervals and standard errors for each from-to party switch across simulations for each alpha and noise combination
        switch_matrix_summary = compute_mean_ci_se(
            df=switch_matrix_df,
            group_cols=["Alpha", "Noise", "From", "To"],
            value_col="Relative",
        )

        # Create dataframe for switch frequencies and compute mean, confidence intervals and standard errors for party switch frequency and candidate switch frequency across simulations for each alpha and noise combination
        switch_frequency_df = pd.DataFrame(switch_frequency_rows)

        party_switch_summary = compute_mean_ci_se(
            df=switch_frequency_df,
            group_cols=["Alpha", "Noise", "Baseline Alpha", "Baseline Noise"],
            value_col="Party Switch Frequency",
        )

        candidate_switch_summary = compute_mean_ci_se(
            df=switch_frequency_df,
            group_cols=["Alpha", "Noise", "Baseline Alpha", "Baseline Noise"],
            value_col="Candidate Switch Frequency",
        )

        # save average relative agreement of top matches to csv (averaged across simulations)
        avg_relative_agreement_df = (
            pd.DataFrame(avg_relative_agreement_rows)
            .groupby(["Alpha", "Noise"], as_index=False)[
                "Avg Relative Agreement (Top Match)"
            ]
            .mean()
        )
        avg_relative_agreement_df.to_csv(
            os.path.join(
                self.experiment_folder,
                "tables/avg_relative_agreement_top_match.csv",
            ),
            index=False,
        )

        # create instance of plotting class and plot results
        plotter = Plotting(
            experiment_folder=self.experiment_folder,
            alpha_values=self.alpha_values,
            noise_values=self.noise_values,
            alpha_noise_simulation_party_match_counts=alpha_noise_simulation_party_match_counts,
            alpha_noise_simulation_candidate_match_counts=alpha_noise_simulation_candidate_match_counts,
            candidates_votes=candidates_votes,
            candidates_per_party=candidates_per_party,
            party_match_summary=party_match_summary,
            election_results=election_results,
            baseline_alpha=self.baseline_alpha,
            sampling_type=self.sample_type,
        )

        for noise in self.noise_values:
            df_filtered = switch_matrix_summary[switch_matrix_summary["Noise"] == noise]
            # create heatmap and sankey diagrams for voter travel
            for alpha in self.alpha_values:
                plotter.voter_travel_heat_map(
                    df_filtered,
                    alpha=alpha,
                    noise=noise,
                )
                plotter.voter_travel_sankey_diagram(
                    df_filtered,
                    alpha,
                    noise,
                )

        # create heatmaps for party switch frequency and candidate switch frequency across alpha and noise combinations with SEM
        for baseline_noise in self.noise_values:
            party_switch_filtered = party_switch_summary[
                party_switch_summary["Baseline Noise"] == baseline_noise
            ]
            candidate_switch_filtered = candidate_switch_summary[
                candidate_switch_summary["Baseline Noise"] == baseline_noise
            ]
            plotter.heat_map_switch_frequency(
                party_switch_filtered,
                switch_type="party",
                baseline_noise=baseline_noise,
            )
            plotter.heat_map_switch_frequency(
                candidate_switch_filtered,
                switch_type="candidate",
                baseline_noise=baseline_noise,
            )

        # create alpha comparison and party distribution plots
        plotter.alpha_comparison()
        plotter.alpha_comparison_diff_candidate_shares()
        plotter.alpha_comparison_diff_vote_shares()
        plotter.alpha_comparison_relative_change()
        plotter.alpha_comparison_relative_change_per_party()
        plotter.alpha_comparison_relative_change_absolute()
        plotter.alpha_comparison_relative_change_per_party_absolute()
        plotter.alpha_comparison_voter_retention_all_parties(switch_matrix_summary)
        plotter.alpha_comparison_voter_retention_per_party(switch_matrix_summary)
        plotter.relative_match_confidence_intervals()
        plotter.party_distributions()
        plotter.party_match_count_boxplots(alpha_noise_simulation_party_match_counts)
        plotter.candidate_matches()
        return None


# run main function with default parameters
if __name__ == "__main__":
    master = MasterFunction()
    master.register_experiment()
    (
        data_q_id,
        election_results,
        candidates_votes,
        candidate_party,
        candidates_per_party,
    ) = master.prepare_data()
    data_mapped_dict = master.map_data(data_q_id)
    master.analysis(
        data_mapped_dict,
        candidate_party,
        election_results,
        candidates_votes,
        candidates_per_party,
    )
