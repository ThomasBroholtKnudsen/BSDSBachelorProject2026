import seaborn as sns
import matplotlib.pyplot as plt
from utils import PARTY_LABEL_MAP, compute_mean_ci_se
import pandas as pd
import json


class Plotting:
    """Generates all match-count and comparison plots for a given experiment."""

    def __init__(
        self,
        experiment_folder,
        alpha_values,
        noise_values,
        alpha_noise_simulation_party_match_counts,
        alpha_noise_simulation_candidate_match_counts,
        candidates_votes,
        candidates_per_party,
        party_match_summary=None,
        election_results=None,
        baseline_alpha=None,
        sampling_type=None,
    ):
        self.experiment_folder = experiment_folder
        self.alpha_values = alpha_values
        self.noise_values = noise_values
        self.alpha_noise_simulation_party_match_counts = (
            alpha_noise_simulation_party_match_counts
        )
        self.alpha_noise_simulation_candidate_match_counts = (
            alpha_noise_simulation_candidate_match_counts
        )
        self.candidates_votes = candidates_votes
        self.candidates_per_party = candidates_per_party.rename(index=PARTY_LABEL_MAP)
        self.party_match_summary = party_match_summary
        self.election_results = election_results
        self.baseline_alpha = baseline_alpha
        self.sampling_type = sampling_type
        self.color_discrete_map = {
            "Socialdemokratiet": "#A82721",
            "Venstre": "#254264",
            "SF": "#E07EA8",
            "Enhedslisten": "#E6801A",
            "Radikale Venstre": "#733280",
            "Dansk Folkeparti": "#EAC73E",
            "Alternativet": "#2B8738",
            "Danmarksdemokraterne": "#7896D2",
            "Konservative": "#96B226",
            "Liberal Alliance": "#3FB2BE",
            "Moderaterne": "#B48CD2",
            "Borgernes Parti": "#5FC8B4",
            "Uden for parti": "#b9b9b9",
        }

    def alpha_comparison(self):
        """Plot relative match counts per party across alpha values for each noise level."""
        print(
            "Creating plot for comparing party match distributions across alpha values..."
        )

        for noise in self.noise_values:
            plt.figure(figsize=(12, 7))

            noise_df = self.party_match_summary[
                self.party_match_summary["Noise"] == noise
            ]
            for party, party_df in noise_df.groupby("Parti"):
                party_df = party_df.sort_values("Alpha")
                plt.plot(
                    party_df["Alpha"],
                    party_df["Mean"],
                    marker="o",
                    label=party,
                    color=self.color_discrete_map.get(party, None),
                )
            plt.ylim(0, noise_df["Mean"].max() * 1.1)

            plt.title(f"Mean Relative Match Counts for Noise = {noise}")
            plt.xlabel("Alpha")
            plt.ylabel("Mean Relative Match Count")
            plt.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                f"{self.experiment_folder}/all_parties_alpha_comparison/alpha_comparison_party_distribution_noise_{noise}.png"
            )
            plt.close()

    def alpha_comparison_diff_candidate_shares(self):
        """Plot each party's match count deviation from its expected share based on candidate counts."""
        print(
            "Creating plot for comparing party match deviation from expected across alpha values..."
        )
        total_candidates = self.candidates_per_party.sum()

        for noise in self.noise_values:
            plt.figure(figsize=(12, 7))

            noise_df = self.party_match_summary[
                self.party_match_summary["Noise"] == noise
            ]

            # collect values for Excel export
            export_rows = []

            for party, party_df in noise_df.groupby("Parti"):
                party_df = party_df.sort_values("Alpha")
                expected = (
                    self.candidates_per_party.get(party, 0) / total_candidates
                    if total_candidates > 0
                    else 0
                )
                label = f"{party} ({expected:.1%})"
                deviation = party_df["Mean"] - expected
                plt.plot(
                    party_df["Alpha"],
                    deviation,
                    marker="o",
                    label=label,
                    color=self.color_discrete_map.get(party, None),
                )

                for alpha, dev in zip(party_df["Alpha"], deviation):
                    export_rows.append({
                        "Parti": party,
                        "Noise": noise,
                        "Alpha": alpha,
                        "Expected Candidate Share": expected,
                        "Mean Relative Count": party_df[party_df["Alpha"] == alpha][
                            "Mean"
                        ].values[0],
                        "Deviation": dev,
                    })

            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.title(
                f"Mean Relative Match Counts Deviation from Expected Fractions (Candidate Shares) for Noise = {noise}"
            )
            plt.xlabel("Alpha")
            plt.ylabel("Mean Relative Match Count − Expected Fraction")
            plt.legend(
                title="Party (candidate shares)",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
            plt.tight_layout()
            plt.savefig(
                f"{self.experiment_folder}/all_parties_alpha_comparison/alpha_comparison_candidate_shares_party_deviation_noise_{noise}.png"
            )
            plt.close()

            # save to Excel
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                f"{self.experiment_folder}/tables/party_match_count_deviations/candidate_shares_deviation_values_noise_{noise}.xlsx",
                index=False,
            )

    def alpha_comparison_diff_vote_shares(self):
        """Plot each party's match count deviation from its expected share based on actual vote shares."""
        print(
            "Creating plot for comparing party match deviation from expected across alpha values..."
        )
        # total_candidates = self.candidates_per_party.sum()

        vote_shares = (
            self.election_results.groupby("Partinavn")[["Stemmetal"]]
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

        vote_shares = vote_shares.rename(index=PARTY_LABEL_MAP)

        vote_share_lookup = vote_shares["Normalized"].to_dict()

        for noise in self.noise_values:
            plt.figure(figsize=(12, 7))

            noise_df = self.party_match_summary[
                self.party_match_summary["Noise"] == noise
            ]

            # collect values for Excel export
            export_rows = []

            for party, party_df in noise_df.groupby("Parti"):
                party_df = party_df.sort_values("Alpha")
                expected = vote_share_lookup.get(party, 0)
                label = f"{party} ({expected:.1%})"
                deviation = party_df["Mean"] - expected
                plt.plot(
                    party_df["Alpha"],
                    deviation,
                    marker="o",
                    label=label,
                    color=self.color_discrete_map.get(party, None),
                )

                for alpha, dev in zip(party_df["Alpha"], deviation):
                    export_rows.append({
                        "Parti": party,
                        "Noise": noise,
                        "Alpha": alpha,
                        "Expected Vote Share": expected,
                        "Mean Relative Count": party_df[party_df["Alpha"] == alpha][
                            "Mean"
                        ].values[0],
                        "Deviation": dev,
                    })

            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.title(
                f"Mean Relative Match Counts Deviation from Expected Fractions (Vote Shares) for Noise = {noise}"
            )
            plt.xlabel("Alpha")
            plt.ylabel("Mean Relative Match Count − Expected Fraction")
            plt.legend(
                title="Party (vote shares)", bbox_to_anchor=(1.02, 1), loc="upper left"
            )
            plt.tight_layout()
            plt.savefig(
                f"{self.experiment_folder}/all_parties_alpha_comparison/alpha_comparison_vote_shares_party_deviation_noise_{noise}.png"
            )
            plt.close()

            # save to Excel
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                f"{self.experiment_folder}/tables/party_match_count_deviations/vote_shares_deviation_values_noise_{noise}.xlsx",
                index=False,
            )

    def alpha_comparison_relative_change(self):
        """Plot each party's relative change in relative match count compared to baseline alpha.
        Uses relative counts. All parties on one plot, no CI."""
        print(
            "Creating plot for comparing relative change in match counts across alpha values..."
        )

        exclude_parties = {"Uden for parti"}

        for noise in self.noise_values:
            plt.figure(figsize=(12, 7))

            export_rows = []
            unique_parties = self.alpha_noise_simulation_party_match_counts[
                "Parti"
            ].unique()

            for party in unique_parties:
                if party in exclude_parties:
                    continue

                baseline_sim_df = self.alpha_noise_simulation_party_match_counts[
                    (self.alpha_noise_simulation_party_match_counts["Parti"] == party)
                    & (self.alpha_noise_simulation_party_match_counts["Noise"] == noise)
                    & (
                        self.alpha_noise_simulation_party_match_counts["Alpha"]
                        == self.baseline_alpha
                    )
                ][["Simulation", "Relative Count"]].rename(
                    columns={"Relative Count": "Baseline Relative Count"}
                )

                if baseline_sim_df.empty:
                    continue

                party_sim_df = self.alpha_noise_simulation_party_match_counts[
                    (self.alpha_noise_simulation_party_match_counts["Parti"] == party)
                    & (self.alpha_noise_simulation_party_match_counts["Noise"] == noise)
                ]

                merged = party_sim_df.merge(baseline_sim_df, on="Simulation")
                merged = merged[merged["Baseline Relative Count"] > 0]

                if merged.empty:
                    continue

                merged["Relative Change"] = (
                    (merged["Relative Count"] - merged["Baseline Relative Count"])
                    / merged["Baseline Relative Count"]
                    * 100
                )

                rc_summary = (
                    compute_mean_ci_se(
                        merged, group_cols=["Alpha"], value_col="Relative Change"
                    )
                    .set_index("Alpha")
                    .sort_index()
                )
                mean_relative_change = rc_summary["Mean"]

                if mean_relative_change.empty:
                    continue

                plt.plot(
                    mean_relative_change.index,
                    mean_relative_change.values,
                    marker="o",
                    label=party,
                    color=self.color_discrete_map.get(party, None),
                )

                for alpha, row in rc_summary.iterrows():
                    export_rows.append({
                        "Parti": party,
                        "Noise": noise,
                        "Alpha": alpha,
                        "Baseline Alpha": self.baseline_alpha,
                        "Mean Relative Change (%)": row["Mean"],
                        "CI Lower (%)": row["Mean"] - row["CI"],
                        "CI Upper (%)": row["Mean"] + row["CI"],
                    })

            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
            )
            plt.title(
                f"Mean Relative Change in Relative Match Counts from Alpha = {self.baseline_alpha} for Noise = {noise}"
            )
            plt.xlabel("Alpha")
            plt.ylabel(f"Change from Baseline Alpha = {self.baseline_alpha}")
            plt.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                f"{self.experiment_folder}/all_parties_alpha_comparison/alpha_comparison_relative_change_relative_match_count_noise_{noise}.png"
            )
            plt.close()

            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                f"{self.experiment_folder}/tables/party_relative_match_count_relative_change/relative_change_values_noise_{noise}.xlsx",
                index=False,
            )

    def alpha_comparison_relative_change_per_party(self):
        """Plot relative change in relative match count with SE bars per party compared to baseline alpha.
        Uses relative counts. Relative change is computed per simulation to correctly propagate uncertainty.
        Baseline alpha is shown as a fixed reference point at 0."""
        print(
            "Creating per-party plots for relative change in relative match counts across alpha values..."
        )

        exclude_parties = {"Uden for parti"}
        unique_parties = self.alpha_noise_simulation_party_match_counts[
            "Parti"
        ].unique()

        for party in unique_parties:
            if party in exclude_parties:
                continue

            for noise in self.noise_values:
                baseline_sim_df = self.alpha_noise_simulation_party_match_counts[
                    (self.alpha_noise_simulation_party_match_counts["Parti"] == party)
                    & (self.alpha_noise_simulation_party_match_counts["Noise"] == noise)
                    & (
                        self.alpha_noise_simulation_party_match_counts["Alpha"]
                        == self.baseline_alpha
                    )
                ][["Simulation", "Relative Count"]].rename(
                    columns={"Relative Count": "Baseline Relative Count"}
                )

                if baseline_sim_df.empty:
                    continue

                party_sim_df = self.alpha_noise_simulation_party_match_counts[
                    (self.alpha_noise_simulation_party_match_counts["Parti"] == party)
                    & (self.alpha_noise_simulation_party_match_counts["Noise"] == noise)
                ]

                merged = party_sim_df.merge(baseline_sim_df, on="Simulation")
                merged = merged[merged["Baseline Relative Count"] > 0]

                if merged.empty:
                    continue

                merged["Relative Change"] = (
                    (merged["Relative Count"] - merged["Baseline Relative Count"])
                    / merged["Baseline Relative Count"]
                    * 100
                )

                summary = compute_mean_ci_se(
                    df=merged,
                    group_cols=["Alpha"],
                    value_col="Relative Change",
                ).sort_values("Alpha")

                if summary.empty:
                    continue

                # Pin baseline alpha to 0 with SE of 0
                summary.loc[summary["Alpha"] == self.baseline_alpha, "Mean"] = 0
                summary.loc[summary["Alpha"] == self.baseline_alpha, "SE"] = 0

                plt.figure(figsize=(12, 7))
                plt.errorbar(
                    summary["Alpha"],
                    summary["Mean"],
                    yerr=summary["SE"],
                    marker="o",
                    color=self.color_discrete_map.get(party, None),
                    label=party,
                    capsize=4,
                    linewidth=1.5,
                )
                # Overlay baseline with star marker
                plt.plot(
                    [self.baseline_alpha],
                    [0],
                    marker="*",
                    markersize=12,
                    color=self.color_discrete_map.get(party, None),
                    label=f"Baseline (alpha={self.baseline_alpha})",
                    linestyle="none",
                )
                plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
                plt.gca().yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
                )
                plt.title(
                    f"Relative Change in Relative Match Count — {party}, noise = {noise}\n"
                    f"Baseline Alpha = {self.baseline_alpha} | Mean ± SE across simulations"
                )
                plt.xlabel("Alpha")
                plt.ylabel(
                    f"Mean Relative Change from Baseline Alpha = {self.baseline_alpha}"
                )
                plt.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/per_party_alpha_comparison/relative_match_count_relative_change_from_baseline/"
                    f"relative_change_party_{party}_noise_{noise}.png"
                )
                plt.close()

    def party_distributions(self):
        """Plot the relative match count distribution across parties for each alpha/noise combination."""
        print("Creating plot for match distribution per party...")

        for alpha in self.alpha_values:
            for noise in self.noise_values:
                plot_df = self.party_match_summary[
                    (self.party_match_summary["Alpha"] == alpha)
                    & (self.party_match_summary["Noise"] == noise)
                ].sort_values("Mean", ascending=False)

                plt.figure(figsize=(10, 6))
                ax = sns.barplot(
                    data=plot_df,
                    x="Parti",
                    y="Mean",
                    errorbar=None,
                )

                # Add SE error bars manually
                for i, row in enumerate(plot_df.itertuples()):
                    ax.errorbar(
                        x=i,
                        y=row.Mean,
                        yerr=row.SE,
                        fmt="none",
                        color="black",
                        capsize=4,
                        linewidth=1.5,
                    )

                # Label bars above the error bar tops
                for i, row in enumerate(plot_df.itertuples()):
                    ax.text(
                        i,
                        row.Mean + row.SE + 0.002,  # just above the error bar top
                        f"{row.Mean:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                plt.xticks(rotation=90)
                plt.title(
                    f"Distribution of Mean Relative Match Counts ± SE for Alpha = {alpha} and Noise = {noise}"
                )
                plt.xlabel("Party")
                plt.ylim(0, (plot_df["Mean"] + plot_df["SE"]).max() * 1.1)
                plt.ylabel("Mean Relative Match Count")

                if self.sampling_type == "relative_to_number_candidates":
                    total_candidates = self.candidates_per_party.sum()
                    legend_title = "Candidate Shares:\n"
                    legend_text = legend_title + "\n".join(
                        f"{party}: {count / total_candidates:.1%}"
                        for party, count in self.candidates_per_party.items()
                    )
                elif self.sampling_type in ["relative", "n_total_relative_to_votes"]:
                    vote_shares = (
                        self.election_results.groupby("Partinavn")[["Stemmetal"]]
                        .sum()
                        .assign(
                            Normalized=lambda x: x["Stemmetal"] / x["Stemmetal"].sum()
                        )
                        .rename(
                            index={
                                "Enhedslisten - De Rød-Grønne": "Enhedslisten – De Rød-Grønne",
                                "Danmarksdemokraterne - Inger Støjberg": "Danmarksdemokraterne ‒ Inger Støjberg",
                                "Venstre": "Venstre, Danmarks Liberale Parti",
                                "(Uden for parti)": "Uden for parti",
                            }
                        )
                        .rename(index=PARTY_LABEL_MAP)
                    )
                    vote_share_lookup = vote_shares["Normalized"].to_dict()
                    legend_title = "Vote Shares:\n"
                    legend_text = legend_title + "\n".join(
                        f"{party}: {vote_share_lookup.get(party, 0):.1%}"
                        for party in self.candidates_per_party.index
                    )
                else:
                    legend_text = "Candidates per Party:\n" + "\n".join(
                        f"{party}: {count}"
                        for party, count in self.candidates_per_party.items()
                    )
                plt.text(
                    0.98,
                    0.98,
                    legend_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/party_distribution_plots/alpha_{alpha}_noise_{noise}_party_distribution.png"
                )
                plt.close()

    def candidate_matches(self):
        """Plot top-10 candidates by mean match count across simulations for each alpha/noise combination."""
        print("Creating plot for candidate match counts (top 10)...")
        for alpha in self.alpha_values:
            for noise in self.noise_values:
                filtered_df = self.alpha_noise_simulation_candidate_match_counts[
                    (
                        self.alpha_noise_simulation_candidate_match_counts["Alpha"]
                        == alpha
                    )
                    & (
                        self.alpha_noise_simulation_candidate_match_counts["Noise"]
                        == noise
                    )
                ]

                # Use compute_mean_ci_se to get mean and SE per candidate
                candidate_summary = (
                    compute_mean_ci_se(
                        df=filtered_df,
                        group_cols=["Kandidat", "Parti"],
                        value_col="Count",
                    )
                    .nlargest(10, "Mean")
                    .sort_values("Mean", ascending=False)
                )

                # Create anonymised labels ranked by mean count
                candidate_summary = candidate_summary.reset_index(drop=True)
                candidate_summary["Anonymised"] = [
                    f"Candidate {i + 1}" for i in range(len(candidate_summary))
                ]

                plt.figure(figsize=(12, 7))
                ax = sns.barplot(
                    data=candidate_summary,
                    x="Anonymised",
                    y="Mean",
                    hue="Parti",
                    palette=self.color_discrete_map,
                    dodge=False,
                    errorbar=None,
                )

                # Add SE error bars manually
                for i, row in enumerate(candidate_summary.itertuples()):
                    ax.errorbar(
                        x=i,
                        y=row.Mean,
                        yerr=row.SE,
                        fmt="none",
                        color="black",
                        capsize=4,
                        linewidth=1.5,
                    )

                # Label bars above error bars
                for i, row in enumerate(candidate_summary.itertuples()):
                    ax.text(
                        i,
                        row.Mean + row.SE + 0.1,
                        f"{row.Mean:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                plt.xticks(rotation=90)
                plt.title(
                    f"Top Candidates by Mean Match Counts ± SE for Alpha = {alpha} and Noise = {noise}"
                )
                plt.xlabel("Candidate")
                plt.ylabel("Mean Match Count")
                plt.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/candidate_match_bar_plots/alpha_{alpha}_noise_{noise}_candidate_matches.png"
                )
                plt.close()

    def relative_match_confidence_intervals(self):
        """Plot mean relative match count with confidence interval band per party across alpha values."""
        print("Creating plot with confidence intervals per party...")

        unique_parties = self.party_match_summary["Parti"].unique()

        for party in unique_parties:
            party_df = self.party_match_summary[
                self.party_match_summary["Parti"] == party
            ]

            for noise in self.noise_values:
                noise_df = party_df[party_df["Noise"] == noise].sort_values("Alpha")

                plt.figure(figsize=(12, 7))
                plt.plot(
                    noise_df["Alpha"],
                    noise_df["Mean"],
                    marker="o",
                    color=self.color_discrete_map.get(party, None),
                    label=party,
                )
                plt.fill_between(
                    noise_df["Alpha"],
                    noise_df["Lower_CI"],
                    noise_df["Upper_CI"],
                    alpha=0.2,
                    color=self.color_discrete_map.get(party, None),
                )
                plt.title(f"Mean Relative Match Count — {party}, Noise = {noise}")
                plt.xlabel("Alpha")
                plt.ylabel("Mean Relative Match Count")
                plt.ylim(0, 0.3)  # noise_df["Upper_CI"].max() * 1.1)
                plt.legend(title="Parti", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/per_party_alpha_comparison/relative_match_counts/{party}_noise_{noise}.png"
                )
                plt.close()

    def alpha_comparison_voter_retention_all_parties(self, switch_matrix_summary):
        """Plot the fraction of voters who stay with their sampled party across alpha values."""
        print("Creating voter retention plot...")

        for noise in self.noise_values:
            plt.figure(figsize=(12, 7))

            noise_df = switch_matrix_summary[switch_matrix_summary["Noise"] == noise]

            # Keep only diagonal values (From == To)
            diagonal_df = noise_df[noise_df["From"] == noise_df["To"]]

            # Collect values for Excel export
            export_rows = []

            for party, party_df in diagonal_df.groupby("From"):
                party_df = party_df.sort_values("Alpha")
                plt.plot(
                    party_df["Alpha"],
                    party_df["Mean"],
                    marker="o",
                    label=party,
                    color=self.color_discrete_map.get(party, None),
                )

                for _, row in party_df.iterrows():
                    export_rows.append({
                        "Parti": party,
                        "Noise": noise,
                        "Alpha": row["Alpha"],
                        "Mean Retention": row["Mean"],
                        "SE": row["SE"],
                        "Lower_CI": row["Lower_CI"],
                        "Upper_CI": row["Upper_CI"],
                        "N": row["N"],
                    })

            plt.title(f"Voter Retention Compared to Baseline for Noise = {noise}")
            plt.xlabel("Alpha")
            plt.ylabel("Fraction of Voters Retained")
            plt.ylim(0, 1)
            plt.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                f"{self.experiment_folder}/all_parties_alpha_comparison/voter_retention_noise_{noise}.png"
            )
            plt.close()

            # Save to Excel
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                f"{self.experiment_folder}/tables/voter_retention/voter_retention_values_noise_{noise}.xlsx",
                index=False,
            )

    def alpha_comparison_voter_retention_per_party(self, switch_matrix_summary):
        """Plot voter retention with CI band per party across alpha values."""
        print("Creating voter retention plots per party...")

        unique_parties = switch_matrix_summary["From"].unique()

        for party in unique_parties:
            for noise in self.noise_values:
                noise_df = switch_matrix_summary[
                    (switch_matrix_summary["Noise"] == noise)
                    & (switch_matrix_summary["From"] == party)
                    & (switch_matrix_summary["To"] == party)
                ].sort_values("Alpha")

                if noise_df.empty:
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(
                    noise_df["Alpha"],
                    noise_df["Mean"],
                    marker="o",
                    color=self.color_discrete_map.get(party, None),
                    label=party,
                )
                plt.fill_between(
                    noise_df["Alpha"],
                    noise_df["Lower_CI"],
                    noise_df["Upper_CI"],
                    alpha=0.2,
                    color=self.color_discrete_map.get(party, None),
                )
                plt.title(f"Voter Retention — {party}, Noise = {noise}")
                plt.xlabel("Alpha")
                plt.ylabel("Fraction of voters retained")
                plt.ylim(0, 1)
                plt.legend(title="Parti", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/per_party_alpha_comparison/voter_retention/{party}_noise_{noise}.png"
                )
                plt.close()

    def party_match_count_boxplots(self, alpha_noise_simulation_party_match_counts):
        """Box plots of relative match count distribution across simulations per party."""
        print("Creating box plots of party match count distributions...")

        for noise in self.noise_values:
            for alpha in self.alpha_values:
                plot_df = alpha_noise_simulation_party_match_counts[
                    (alpha_noise_simulation_party_match_counts["Noise"] == noise)
                    & (alpha_noise_simulation_party_match_counts["Alpha"] == alpha)
                ].sort_values("Relative Count", ascending=False)

                plt.figure(figsize=(12, 7))
                ax = plt.gca()

                # Get unique parties in order of median relative count
                party_order = (
                    plot_df.groupby("Parti")["Relative Count"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist()
                )

                # Draw one box per party with party colour
                box_data = [
                    plot_df[plot_df["Parti"] == party]["Relative Count"].values
                    for party in party_order
                ]

                bp = ax.boxplot(
                    box_data,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                )

                for patch, party in zip(bp["boxes"], party_order):
                    color = self.color_discrete_map.get(party, "#888888")
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticks(range(1, len(party_order) + 1))
                ax.set_xticklabels(party_order, rotation=90, fontsize=9)
                ax.set_title(
                    f"Distribution of Mean Relative Match Counts for Alpha {alpha} and Noise {noise}"
                )
                ax.set_xlabel("Party")
                ax.set_ylabel("Mean Relative Match Count")
                ax.set_ylim(0, plot_df["Relative Count"].max() * 1.1)

                plt.tight_layout()
                plt.savefig(
                    f"{self.experiment_folder}/party_distribution_plots/"
                    f"boxplot_alpha_{alpha}_noise_{noise}.png"
                )
                plt.close()

    def voter_travel_heat_map(self, switch_matrix_summary, alpha=None, noise=None):
        """Plot and save a party-switch heatmap with mean ± SE annotations across simulations.

        Args:
            switch_matrix_summary: DataFrame with columns From, To, Alpha, Noise, Mean, SE
            experiment_folder: root output directory
            alpha: alpha value to filter and label
            noise: noise level to filter and label
        """
        df = switch_matrix_summary[
            (switch_matrix_summary["Alpha"] == alpha)
            & (switch_matrix_summary["Noise"] == noise)
        ]

        # Mean matrix — already normalized per simulation so no need to normalize again
        matrix_mean = df.pivot_table(
            index="From", columns="To", values="Mean", aggfunc="sum", fill_value=0
        )

        # SE matrix
        matrix_se = df.pivot_table(
            index="From", columns="To", values="SE", aggfunc="sum", fill_value=0
        )

        # Annotation matrix
        annot = pd.DataFrame(
            index=matrix_mean.index, columns=matrix_mean.columns, dtype=str
        )
        for row_idx in matrix_mean.index:
            for col_idx in matrix_mean.columns:
                mean_val = matrix_mean.loc[row_idx, col_idx]
                se_val = (
                    matrix_se.loc[row_idx, col_idx]
                    if row_idx in matrix_se.index and col_idx in matrix_se.columns
                    else 0
                )
                annot.loc[row_idx, col_idx] = (
                    f"{mean_val * 100:.0f}%\n±{se_val * 100:.1f}%"
                )

        mask_diag = pd.DataFrame(
            False, index=matrix_mean.index, columns=matrix_mean.columns
        )
        for p in matrix_mean.index:
            if p in matrix_mean.columns:
                mask_diag.loc[p, p] = True

        _, ax = plt.subplots(figsize=(14, 11))

        sns.heatmap(
            matrix_mean,
            annot=annot,
            fmt="",
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 7},
            cbar_kws={"label": "Fraction of voters", "shrink": 0.8},
            ax=ax,
            vmin=0,
            vmax=1,
        )

        sns.heatmap(
            matrix_mean,
            mask=~mask_diag,
            annot=annot,
            fmt="",
            cmap="Greys",
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 7},
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=1,
        )

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        colorbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        colorbar.set_label("Percentage of voters")

        ax.set_title(
            f"Voter Retention & Switching (Mean ± SE Across Simulations)\n"
            f"Alpha = {alpha}, Noise = {noise}",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        ax.set_xlabel("New matched party", fontsize=11, labelpad=10)
        ax.set_ylabel("Original sampled party", fontsize=11, labelpad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.tick_params(axis="y", rotation=0, labelsize=8)

        plt.tight_layout()
        plt.savefig(
            f"{self.experiment_folder}/voter_travel_plots/voter_travel_heatmaps/"
            f"alpha_{alpha}_noise_{noise}_heatmap.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def heat_map_switch_frequency(
        self, switch_summary, switch_type="party", baseline_noise=None
    ):
        title = (
            f"Switching Frequencies — Parties (Mean ± SE Across Simulations)\n"
            f"Baseline: alpha={self.baseline_alpha}, noise={baseline_noise}"
            if switch_type == "party"
            else f"Switching Frequencies — Candidates (Mean ± SE Across Simulations)\n"
            f"Baseline: alpha={self.baseline_alpha}, noise={baseline_noise}"
        )
        filename = (
            f"party_switching_frequencies_baseline_alpha_{self.baseline_alpha}_noise_{baseline_noise}.png"
            if switch_type == "party"
            else f"candidate_switching_frequencies_baseline_alpha_{self.baseline_alpha}_noise_{baseline_noise}.png"
        )

        # Get all unique alpha and noise values
        all_alphas = sorted(switch_summary["Alpha"].unique())
        all_noises = sorted(switch_summary["Noise"].unique())

        # Build cross-shaped index:
        # - all alphas at baseline noise (the row)
        # - all noises at baseline alpha (the column)
        cross_combinations = set()
        for alpha in all_alphas:
            cross_combinations.add((alpha, baseline_noise))  # baseline noise row
        for noise in all_noises:
            cross_combinations.add((
                self.baseline_alpha,
                noise,
            ))  # baseline alpha column

        # Filter switch_summary to only cross combinations
        cross_df = switch_summary[
            switch_summary.apply(
                lambda row: (row["Alpha"], row["Noise"]) in cross_combinations, axis=1
            )
        ]

        # Build mean and SE matrices
        mean_matrix = cross_df.pivot(
            index="Noise", columns="Alpha", values="Mean"
        ).sort_index(ascending=True)  # small noise at bottom

        se_matrix = cross_df.pivot(
            index="Noise", columns="Alpha", values="SE"
        ).sort_index(ascending=True)

        # Sort columns ascending (small alpha on left)
        mean_matrix = mean_matrix.sort_index(axis=1, ascending=True)
        se_matrix = se_matrix.sort_index(axis=1, ascending=True)

        # Annotation matrix
        annot = pd.DataFrame(
            index=mean_matrix.index, columns=mean_matrix.columns, dtype=str
        )
        for noise in mean_matrix.index:
            for alpha in mean_matrix.columns:
                if pd.isna(mean_matrix.loc[noise, alpha]):
                    annot.loc[noise, alpha] = ""
                else:
                    mean_val = mean_matrix.loc[noise, alpha]
                    se_val = se_matrix.loc[noise, alpha]
                    annot.loc[noise, alpha] = f"{mean_val:.2f}\n±{se_val:.2f}"

        # Mask NaN cells (not part of the cross)
        mask = mean_matrix.isna()

        _, ax = plt.subplots(figsize=(14, 11))

        sns.heatmap(
            mean_matrix,
            annot=annot,
            fmt="",
            mask=mask,
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 8},
            cbar_kws={"label": "Fraction of voters", "shrink": 0.8},
            ax=ax,
            vmin=0,
            vmax=1,
        )

        # Highlight the baseline cell
        baseline_row = mean_matrix.index.get_loc(baseline_noise)
        baseline_col = mean_matrix.columns.get_loc(self.baseline_alpha)
        ax.add_patch(
            plt.Rectangle(
                (baseline_col, baseline_row), 1, 1, fill=False, edgecolor="red", lw=2
            )
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("Alpha", fontsize=11, labelpad=10)
        ax.set_ylabel("Noise", fontsize=11, labelpad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.tick_params(axis="y", rotation=0, labelsize=8)

        # Invert y-axis so small noise is at bottom
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(
            f"{self.experiment_folder}/switch_frequency_plots/{filename}",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def voter_travel_sankey_diagram(
        self,
        switch_matrix_summary,
        alpha,
        noise,
    ):
        """Build and save an interactive HTML Sankey diagram of voter party flow averaged across simulations.

        Args:
            switch_matrix_summary: DataFrame with columns From, To, Alpha, Noise, Mean, SE
            alpha: alpha value to filter and label
            noise: noise level to filter and label
            experiment_folder: root output directory
        """
        color_discrete_map = {
            "Socialdemokratiet": "#A82721",
            "Venstre": "#254264",
            "SF": "#E07EA8",
            "Enhedslisten": "#E6801A",
            "Radikale Venstre": "#733280",
            "Dansk Folkeparti": "#EAC73E",
            "Alternativet": "#2B8738",
            "Danmarksdemokraterne": "#7896D2",
            "Konservative": "#96B226",
            "Liberal Alliance": "#3FB2BE",
            "Moderaterne": "#B48CD2",
            "Borgernes Parti": "#5FC8B4",
            "Uden for parti": "#b9b9b9",
        }

        # Filter to this alpha and noise
        df = switch_matrix_summary[
            (switch_matrix_summary["Alpha"] == alpha)
            & (switch_matrix_summary["Noise"] == noise)
        ]

        if df.empty:
            return None

        # Build rows similar to original but using Mean Weight
        rows = []
        for from_party, group in df.groupby("From"):
            for _, row in group.iterrows():
                rows.append({
                    "from_party": row["From"],
                    "to_party": row["To"],
                    "count": row["Mean"],  # used for sizing
                    "relative": row["Mean"],  # mean of per-simulation relatives
                    "se": row["SE"],  # SE of per-simulation relatives
                })

        voter_travel = pd.DataFrame(rows)
        if voter_travel.empty:
            return None

        left_parties = sorted(voter_travel["from_party"].unique().tolist())

        data_json = json.dumps(voter_travel.to_dict(orient="records"))
        color_map_json = json.dumps(color_discrete_map)
        left_parties_json = json.dumps(left_parties)

        html = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    body {{ font-family: sans-serif; background: white; margin: 20px; }}
    h2 {{ font-size: 16px; font-weight: bold; margin-bottom: 12px; }}
    #controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .party-btn {{
        padding: 5px 12px; border: 2px solid #ccc; border-radius: 4px;
        cursor: pointer; font-size: 12px; background: white;
        transition: all 0.2s; white-space: nowrap;
    }}
    .party-btn.active {{ color: white; }}
    .link {{ fill: none; }}
    .node-label {{ font-size: 13px; dominant-baseline: middle; }}
    .pct-label {{ font-size: 13px; dominant-baseline: middle; font-weight: bold; }}
    .tooltip {{
        position: fixed; background: rgba(0,0,0,0.75); color: white;
        padding: 6px 10px; border-radius: 4px; font-size: 12px;
        pointer-events: none; display: none; z-index: 10;
    }}
    </style>
    </head>
    <body>
    <h2>Voter Flow (mean across simulations) — alpha = {alpha}, noise = {noise}</h2>
    <div id="controls"></div>
    <div class="tooltip" id="tooltip"></div>
    <svg id="chart"></svg>

    <script>
    const data = {data_json};
    const leftParties = {left_parties_json};
    const colorMap = {color_map_json};

    function getColor(p) {{ return colorMap[p] || "#888"; }}

    const svgEl = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");

    const width = 900;
    const nodeH = 28, pad = 16;
    const leftX = 220, rightX = width - 220;

    let activeParty = null;

    function getRightOrder(party) {{
    const totals = {{}};
    data.forEach(d => {{
        if (!party || d.from_party === party) {{
        totals[d.to_party] = (totals[d.to_party] || 0) + d.count;
        }}
    }});
    return Object.entries(totals)
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0]);
    }}

    function getYCenters(n, height) {{
    const totalH = n * nodeH + (n - 1) * pad;
    const startY = (height - totalH) / 2;
    return Array.from({{length: n}}, (_, i) => startY + i * (nodeH + pad) + nodeH / 2);
    }}

    function getRightPct(party, toParty) {{
    const relevant = data.filter(d => !party || d.from_party === party);
    const total = relevant.reduce((s, d) => s + d.count, 0);
    const sub = relevant.filter(d => d.to_party === toParty).reduce((s, d) => s + d.count, 0);
    if (total === 0) return "0%";
    const pct = sub / total * 100;
    return (pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)) + "%";
    }}

    function draw() {{
    svgEl.innerHTML = "";
    const rightOrder = getRightOrder(activeParty);
    const nLeft = leftParties.length;
    const nRight = rightOrder.length;
    const height = Math.max(nLeft, nRight) * (nodeH + pad) + 60;
    svgEl.setAttribute("width", width);
    svgEl.setAttribute("height", height);
    svgEl.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

    const leftYArr = getYCenters(nLeft, height);
    const rightYArr = getYCenters(nRight, height);
    const leftYMap = Object.fromEntries(leftParties.map((p, i) => [p, leftYArr[i]]));
    const rightYMap = Object.fromEntries(rightOrder.map((p, i) => [p, rightYArr[i]]));

    // Links
    data.forEach(d => {{
        const sy = leftYMap[d.from_party];
        const ty = rightYMap[d.to_party];
        if (sy === undefined || ty === undefined) return;
        if (activeParty && d.from_party !== activeParty) return;
        const strokeW = Math.max(1, d.relative * nodeH);
        const color = "#aaaaaa";
        const cx = (leftX + rightX) / 2;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M${{leftX}},${{sy}} C${{cx}},${{sy}} ${{cx}},${{ty}} ${{rightX}},${{ty}}`);
        path.setAttribute("class", "link");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", strokeW);
        path.setAttribute("opacity", "0.6");

        path.addEventListener("mousemove", e => {{
        const pct = (d.relative * 100).toFixed(1);
        const se = (d.se * 100).toFixed(1);
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY - 28) + "px";
        tooltip.textContent = `${{d.from_party}} → ${{d.to_party}}: ${{pct}}% (±${{se}}% SE)`;
        }});
        path.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
        svgEl.appendChild(path);
    }});

    // Left nodes
    leftParties.forEach((p, i) => {{
        const cy = leftYArr[i];
        const color = getColor(p);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", leftX - 14);
        rect.setAttribute("y", cy - nodeH / 2);
        rect.setAttribute("width", 10);
        rect.setAttribute("height", nodeH);
        rect.setAttribute("fill", color);
        svgEl.appendChild(rect);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", leftX - 20);
        label.setAttribute("y", cy);
        label.setAttribute("text-anchor", "end");
        label.setAttribute("class", "node-label");
        label.setAttribute("font-weight", activeParty === p ? "bold" : "normal");
        label.textContent = p;
        label.addEventListener("mousemove", e => {{
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY - 28) + "px";
        tooltip.textContent = `${{p}}`;
        }});
        label.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
        svgEl.appendChild(label);
    }});

    // Right nodes
    rightOrder.forEach((p, i) => {{
        const cy = rightYArr[i];
        const color = getColor(p);
        const pct = getRightPct(activeParty, p);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", rightX + 4);
        rect.setAttribute("y", cy - nodeH / 2);
        rect.setAttribute("width", 10);
        rect.setAttribute("height", nodeH);
        rect.setAttribute("fill", color);
        svgEl.appendChild(rect);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", rightX + 20);
        label.setAttribute("y", cy);
        label.setAttribute("text-anchor", "start");
        label.setAttribute("class", "node-label");
        label.textContent = p;
        label.addEventListener("mousemove", e => {{
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY - 28) + "px";
        tooltip.textContent = p;
        }});
        label.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
        svgEl.appendChild(label);

        const pctLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        pctLabel.setAttribute("x", width - 10);
        pctLabel.setAttribute("y", cy);
        pctLabel.setAttribute("text-anchor", "end");
        pctLabel.setAttribute("class", "pct-label");
        pctLabel.textContent = pct;
        svgEl.appendChild(pctLabel);
    }});
    }}

    // Buttons
    const controls = document.getElementById("controls");
    leftParties.forEach(p => {{
    const btn = document.createElement("button");
    btn.textContent = p;
    btn.className = "party-btn";
    btn.style.borderColor = getColor(p);
    btn.onclick = () => {{
        if (activeParty === p) {{
        activeParty = null;
        btn.classList.remove("active");
        btn.style.background = "white";
        btn.style.color = "black";
        }} else {{
        activeParty = p;
        document.querySelectorAll(".party-btn").forEach(b => {{
            b.classList.remove("active");
            b.style.background = "white";
            b.style.color = "black";
        }});
        btn.classList.add("active");
        btn.style.background = getColor(p);
        btn.style.color = "white";
        }}
        draw();
    }};
    controls.appendChild(btn);
    }});

    draw();
    </script>
    </body>
    </html>"""

        output_path = (
            f"{self.experiment_folder}/voter_travel_plots/voter_travel_sankey_diagrams/"
            f"voter_travel_noise_{noise}_alpha_{alpha}.html"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return None
