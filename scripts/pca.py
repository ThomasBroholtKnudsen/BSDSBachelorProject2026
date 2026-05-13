from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class PCAAnalysis:
    """PCA computation and 2D/3D plotting for candidate and voter response data."""

    def __init__(self, candidate_data, experiment_folder):
        self.candidate_data = candidate_data
        self.experiment_folder = experiment_folder

    def perform_pca(self):
        """Fit PCA on standardized candidate responses.

        Returns:
            Tuple of (pca_components, party_labels, explained_variance_ratio[:3], candidate_names).
        """
        candidate_data = self.candidate_data.copy()

        # Create pivot table: rows = candidates, columns = questions
        fv_pca_pivot = candidate_data.pivot_table(
            index="Navn", columns="Spørgsmål", values="Svar_mapped", aggfunc="first"
        )

        # drop candidates with any missing values
        fv_pca_pivot = fv_pca_pivot.dropna()

        # Standardize the data
        scaler = StandardScaler()
        fv_pca_scaled = scaler.fit_transform(fv_pca_pivot)

        # Perform PCA
        pca = PCA()
        fv_pca_components = pca.fit_transform(fv_pca_scaled)

        # Get candidate names
        # Get party information for each candidate
        candidate_names = list(fv_pca_pivot.index)

        parties = []
        for name in candidate_names:
            party_name = candidate_data[candidate_data["Navn"] == name]["Parti"].iloc[0]
            parties.append(party_name)
        return (
            fv_pca_components,
            parties,
            pca.explained_variance_ratio_[:3],
            candidate_names,
        )

    def pca_plots_candidates(
        self,
        pca_components,
        parties,
        explained_variance_ratio,
        alpha,
        candidate_names=None,
        highlight_name=None,
    ):
        """Plot and save a 2D PCA scatter of candidates coloured by party, with party means overlaid.

        Args:
            pca_components: array of shape (n_candidates, n_components) from perform_pca
            parties: list of party label strings aligned with pca_components rows
            explained_variance_ratio: first 3 explained variance ratios from perform_pca
            alpha: alpha value used in the run (for title and filename)
            candidate_names: list of candidate name strings; required for highlight_name
            highlight_name: if given, marks and labels this candidate with a star
        """
        color_discrete_map = {
            "Socialdemokratiet": "#A82721",
            "Venstre, Danmarks Liberale Parti": "#254264",
            "SF - Socialistisk Folkeparti": "#E07EA8",
            "Enhedslisten – De Rød-Grønne": "#E6801A",
            "Radikale Venstre": "#733280",
            "Dansk Folkeparti": "#EAC73E",
            "Alternativet": "#2B8738",
            "Danmarksdemokraterne ‒ Inger Støjberg": "#7896D2",
            "Det Konservative Folkeparti": "#96B226",
            "Liberal Alliance": "#3FB2BE",
            "Moderaterne": "#B48CD2",
            "Borgernes Parti - Lars Boje Mathiesen": "#5FC8B4",
            "Uden for parti": "#b9b9b9",
        }
        # Plot candidates in first two PCA components colored by party

        # Plot
        point_colors = [color_discrete_map.get(p, "#888888") for p in parties]
        plt.figure(figsize=(14, 10))
        plt.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            color=point_colors,
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Calculate and plot party means
        pca_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "Party": parties,
        })

        party_means_pca = pca_df.groupby("Party")[["PC1", "PC2"]].mean()

        unique_parties = list(set(parties))

        for party in unique_parties:
            mean_pc1 = party_means_pca.loc[party, "PC1"]
            mean_pc2 = party_means_pca.loc[party, "PC2"]
            plt.scatter(
                mean_pc1,
                mean_pc2,
                color=color_discrete_map.get(party, "#888888"),
                s=500,
                alpha=1.0,
                edgecolors="black",
                linewidth=2.5,
                marker="o",
                zorder=5,
            )

        # Highlight specific candidate if requested
        if highlight_name is not None:
            if highlight_name in candidate_names:
                idx = candidate_names.index(highlight_name)
                plt.scatter(
                    pca_components[idx, 0],
                    pca_components[idx, 1],
                    c="white",
                    s=400,
                    alpha=1.0,
                    edgecolors="red",
                    linewidth=3,
                    marker="*",
                    zorder=10,
                )
                plt.annotate(
                    highlight_name,
                    (pca_components[idx, 0], pca_components[idx, 1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=11,
                    fontweight="bold",
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    zorder=11,
                )
            else:
                print(f"Candidate '{highlight_name}' not found in the data.")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_discrete_map[party],
                markersize=10,
                label=party,
            )
            for party in unique_parties
        ]
        plt.legend(handles=legend_elements, loc="best", fontsize=8, framealpha=0.9)

        plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
        plt.title(f"PCA - Candidates by Party (PC1 vs PC2) for Alpha = {alpha}")
        plt.xlim(-8, 8)
        plt.ylim(-7, 7)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.experiment_folder}/pca_candidate_plots/pca_candidates_alpha_{alpha}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved PCA candidate plot for alpha={alpha}")

        return None

    def pca_plots_voters(self, voter_data, alpha, noise, highlight_name=None):
        """Plot and save a 2D PCA scatter of candidates and voters, with voters projected onto the candidate PCA space.

        Args:
            voter_data: long-format voter DataFrame with 'Svar_mapped' and 'noise_id' columns
            alpha: alpha value used in the run (for filename)
            noise: noise level used in the run (for filename)
            highlight_name: if given, marks and labels this candidate or voter with a star
        """
        candidate_data = self.candidate_data.copy()

        # Use the latest available noise iteration instead of a hardcoded id.
        if "noise_id" in voter_data.columns and not voter_data.empty:
            latest_noise_id = voter_data["noise_id"].max()
            voter_data = voter_data[voter_data["noise_id"] == latest_noise_id]

        # Create pivot table: rows = candidates, columns = questions
        pivot_candidates = candidate_data.pivot_table(
            index="Navn",
            columns="Spørgsmål",
            values="Svar_mapped",
            aggfunc="first",
        )

        # Create pivot table: rows = voters, columns = questions
        pivot_voters = voter_data.pivot_table(
            index=["Navn", "sample_id"],
            columns="Spørgsmål",
            values="Svar_mapped",
            aggfunc="first",
        )

        # Remove rows with missing values
        pivot_candidates = pivot_candidates.dropna()
        pivot_voters = pivot_voters.dropna()

        # Ensure both datasets use the same question columns in the same order
        common_questions = pivot_candidates.columns.intersection(pivot_voters.columns)
        pivot_candidates = pivot_candidates[common_questions]
        pivot_voters = pivot_voters[common_questions]

        # Standardize the data
        scaler = StandardScaler()
        candidates_scaled = scaler.fit_transform(pivot_candidates)
        voters_scaled = scaler.transform(pivot_voters)

        candidate_names = pivot_candidates.index.tolist()
        voter_names = pivot_voters.index.tolist()

        # Perform PCA
        pca = PCA()
        pca_components = pca.fit_transform(candidates_scaled)

        # transform voters using the same PCA
        voters_pca = pca.transform(voters_scaled)

        # Plot candidates and voters in first two PCA components colored by voter vs candidate
        plt.figure(figsize=(14, 10))
        plt.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            color="blue",
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label="Candidates",
        )

        plt.scatter(
            voters_pca[:, 0],
            voters_pca[:, 1],
            c="red",
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label="Voters",
        )

        # for i, name in enumerate(candidate_names):
        #     plt.annotate(
        #         name,
        #         (pca_components[i, 0], pca_components[i, 1]),
        #         xytext=(4, 4),
        #         textcoords="offset points",
        #         fontsize=10,
        #         color="blue",
        #         alpha=0.9,
        #     )

        # for i, name in enumerate(voter_names):
        #     plt.annotate(
        #         name,
        #         (voters_pca[i, 0], voters_pca[i, 1]),
        #         xytext=(4, 4),
        #         textcoords="offset points",
        #         fontsize=14,
        #         color="red",
        #         alpha=0.9,
        #     )

        # Highlight specific point if requested (can be candidate or voter)
        if highlight_name is not None:
            if highlight_name in candidate_names:
                idx = candidate_names.index(highlight_name)
                plt.scatter(
                    pca_components[idx, 0],
                    pca_components[idx, 1],
                    c="white",
                    s=600,
                    alpha=1.0,
                    edgecolors="blue",
                    linewidth=3,
                    marker="*",
                    zorder=10,
                )
                plt.annotate(
                    f"{highlight_name} (C)",
                    (pca_components[idx, 0], pca_components[idx, 1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=11,
                    fontweight="bold",
                    color="blue",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                    zorder=11,
                )
            elif highlight_name in voter_names:
                idx = voter_names.index(highlight_name)
                plt.scatter(
                    voters_pca[idx, 0],
                    voters_pca[idx, 1],
                    c="lime",
                    s=400,
                    alpha=1.0,
                    edgecolors="red",
                    linewidth=3,
                    marker="*",
                    zorder=10,
                )
                plt.annotate(
                    f"{highlight_name} (V)",
                    (voters_pca[idx, 0], voters_pca[idx, 1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=14,
                    fontweight="bold",
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    zorder=11,
                )
            else:
                print(f"Person '{highlight_name}' not found in the data.")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Candidates",
                markerfacecolor="blue",
                markersize=22,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Voters",
                markerfacecolor="red",
                markersize=22,
            ),
        ]
        plt.legend(handles=legend_elements, loc="best", fontsize=22, framealpha=0.9)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=16)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=16)
        plt.title(
            f"PCA - Candidates vs Voters (PC1 vs PC2) for Alpha = {alpha} and Noise = {noise}",
            fontsize=20,
        )
        plt.xlim(-8, 8)
        plt.ylim(-7, 7)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.experiment_folder}/pca_voter_plots/pca_voters_alpha_{alpha}_noise_{noise}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved PCA voter plot for alpha={alpha} and noise={noise}")

        return None

    def pca_plots_candidates_3d(
        self,
        pca_components,
        parties,
        explained_variance_ratio,
        alpha,
        candidate_names=None,
        highlight_name=None,
    ):
        """Plot and save an interactive 3D PCA scatter of candidates coloured by party.

        Args:
            pca_components: array of shape (n_candidates, n_components) from perform_pca
            parties: list of party label strings aligned with pca_components rows
            explained_variance_ratio: first 3 explained variance ratios from perform_pca
            alpha: alpha value used in the run (for title and filename)
            candidate_names: list of candidate name strings; required for highlight_name
            highlight_name: if given, marks and labels this candidate with a diamond marker
        """
        color_discrete_map = {
            "Socialdemokratiet": "#A82721",
            "Venstre, Danmarks Liberale Parti": "#254264",
            "SF - Socialistisk Folkeparti": "#E07EA8",
            "Enhedslisten – De Rød-Grønne": "#E6801A",
            "Radikale Venstre": "#733280",
            "Dansk Folkeparti": "#EAC73E",
            "Alternativet": "#2B8738",
            "Danmarksdemokraterne ‒ Inger Støjberg": "#7896D2",
            "Det Konservative Folkeparti": "#96B226",
            "Liberal Alliance": "#3FB2BE",
            "Moderaterne": "#B48CD2",
            "Borgernes Parti - Lars Boje Mathiesen": "#5FC8B4",
            "Uden for parti": "#b9b9b9",
        }

        names = candidate_names if candidate_names is not None else [""] * len(parties)
        pca_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "PC3": pca_components[:, 2],
            "Party": parties,
            "Name": names,
        })

        fig = px.scatter_3d(
            pca_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Party",
            hover_name="Name",
            color_discrete_map=color_discrete_map,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=3))

        # Plot party means as larger markers
        party_means = (
            pca_df.groupby("Party")[["PC1", "PC2", "PC3"]].mean().reset_index()
        )
        for _, row in party_means.iterrows():
            color = color_discrete_map.get(row["Party"], "#888888")
            fig.add_trace(
                go.Scatter3d(
                    x=[row["PC1"]],
                    y=[row["PC2"]],
                    z=[row["PC3"]],
                    mode="markers",
                    marker=dict(size=8, color=color, line=dict(color="black", width=2)),
                    name=f"{row['Party']} (mean)",
                    showlegend=False,
                    hovertemplate=f"{row['Party']} mean<extra></extra>",
                )
            )

        # Highlight specific candidate if requested
        if highlight_name is not None:
            if highlight_name in names:
                row = pca_df[pca_df["Name"] == highlight_name].iloc[0]
                fig.add_trace(
                    go.Scatter3d(
                        x=[row["PC1"]],
                        y=[row["PC2"]],
                        z=[row["PC3"]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color="white",
                            symbol="diamond",
                            line=dict(color="red", width=3),
                        ),
                        text=[highlight_name],
                        textposition="top center",
                        name=highlight_name,
                        showlegend=True,
                    )
                )
            else:
                print(f"Candidate '{highlight_name}' not found in the data.")

        fig.update_layout(
            title=f"PCA - Candidates by Party (3D) for Alpha = {alpha}",
            scene=dict(
                xaxis_title=f"PC1 ({explained_variance_ratio[0]:.1%} variance)",
                yaxis_title=f"PC2 ({explained_variance_ratio[1]:.1%} variance)",
                zaxis_title=f"PC3 ({explained_variance_ratio[2]:.1%} variance)",
            ),
        )

        fig.write_html(
            f"{self.experiment_folder}/pca_candidate_plots/pca_candidates_3d_alpha_{alpha}.html",
        )
        print(f"Saved 3D PCA candidate plot for alpha={alpha}")

        return None

    def pca_plots_voters_3d(self, voter_data, alpha, noise, highlight_name=None):
        """Plot and save an interactive 3D PCA scatter of candidates and voters.

        Args:
            voter_data: long-format voter DataFrame with 'Svar_mapped' and 'noise_id' columns
            alpha: alpha value used in the run (for filename)
            noise: noise level used in the run (for filename)
            highlight_name: if given, marks and labels this candidate or voter with a diamond marker
        """
        candidate_data = self.candidate_data.copy()

        if "noise_id" in voter_data.columns and not voter_data.empty:
            latest_noise_id = voter_data["noise_id"].max()
            voter_data = voter_data[voter_data["noise_id"] == latest_noise_id]

        pivot_candidates = candidate_data.pivot_table(
            index="Navn",
            columns="Spørgsmål",
            values="Svar_mapped",
            aggfunc="first",
        )
        pivot_voters = voter_data.pivot_table(
            index=["Navn", "sample_id"],
            columns="Spørgsmål",
            values="Svar_mapped",
            aggfunc="first",
        )

        pivot_candidates = pivot_candidates.dropna()
        pivot_voters = pivot_voters.dropna()

        common_questions = pivot_candidates.columns.intersection(pivot_voters.columns)
        pivot_candidates = pivot_candidates[common_questions]
        pivot_voters = pivot_voters[common_questions]

        scaler = StandardScaler()
        candidates_scaled = scaler.fit_transform(pivot_candidates)
        voters_scaled = scaler.transform(pivot_voters)

        candidate_names = pivot_candidates.index.tolist()
        voter_names = pivot_voters.index.tolist()

        pca = PCA()
        pca_components = pca.fit_transform(candidates_scaled)
        voters_pca = pca.transform(voters_scaled)

        candidates_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "PC3": pca_components[:, 2],
            "Name": candidate_names,
            "Type": "Candidate",
        })
        voters_df = pd.DataFrame({
            "PC1": voters_pca[:, 0],
            "PC2": voters_pca[:, 1],
            "PC3": voters_pca[:, 2],
            "Name": voter_names,
            "Type": "Voter",
        })
        combined_df = pd.concat([candidates_df, voters_df], ignore_index=True)

        fig = px.scatter_3d(
            combined_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Type",
            hover_name="Name",
            color_discrete_map={"Candidate": "blue", "Voter": "red"},
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=3))

        if highlight_name is not None:
            if highlight_name in candidate_names:
                row = candidates_df[candidates_df["Name"] == highlight_name].iloc[0]
                fig.add_trace(
                    go.Scatter3d(
                        x=[row["PC1"]],
                        y=[row["PC2"]],
                        z=[row["PC3"]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color="white",
                            symbol="diamond",
                            line=dict(color="blue", width=3),
                        ),
                        text=[f"{highlight_name} (C)"],
                        textposition="top center",
                        name=highlight_name,
                        showlegend=True,
                    )
                )
            elif highlight_name in voter_names:
                row = voters_df[voters_df["Name"] == highlight_name].iloc[0]
                fig.add_trace(
                    go.Scatter3d(
                        x=[row["PC1"]],
                        y=[row["PC2"]],
                        z=[row["PC3"]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color="lime",
                            symbol="diamond",
                            line=dict(color="red", width=3),
                        ),
                        text=[f"{highlight_name} (V)"],
                        textposition="top center",
                        name=highlight_name,
                        showlegend=True,
                    )
                )
            else:
                print(f"Person '{highlight_name}' not found in the data.")

        fig.update_layout(
            title=f"PCA - Candidates vs Voters (3D) for Alpha = {alpha} and Noise = {noise}",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
            ),
        )

        fig.write_html(
            f"{self.experiment_folder}/pca_voter_plots/pca_voters_3d_alpha_{alpha}_noise_{noise}.html",
        )
        print(f"Saved 3D PCA voter plot for alpha={alpha} and noise={noise}")

        return None
