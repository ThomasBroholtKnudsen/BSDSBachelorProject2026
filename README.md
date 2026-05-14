# How do you distinguish between "Agree" and "Strongly Agree"? A study on the effect of scaling the extreme response options in a Voting Advice Application

A repository containing the code used for the bachelor thesis presented by Thomas Broholt Knudsen and Christian Høst-Madsen on the 6th semester in Data Science at the IT-University of Copenhagen.

---

## Getting Started

### 1) Clone the repository and enter the project folder
```bash
git clone https://github.com/ThomasBroholtKnudsen/BSDSBachelorProject2026.git
cd BSDSBachelorProject2026
```

### 2) Install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

### 3) Obtain data
Place VAA data in data/raw. Contact the authors or Altinget to obtain the data.

### 4) Configure parameters
Edit the parameters in `scripts/main.py` — specifically the `MasterFunction` constructor arguments (e.g. `alpha_values`, `noise_values`, `sample_type`, `simulations`, etc.). See the research paper for the specific parameter values used. `alpha_values` and `noise_values` must be lists. 

### 5) Run the pipeline
```bash
cd scripts
python main.py
```

Output plots and tables are written to a new folder under `experiments/`.

---

## Project Structure
```text
project_root/
├─ data/
│  ├─ interim/                          # processed/mapped data files (.csv, .xlsx)
│  │     └─ fv_results_26.csv
│  │
│  └─ raw/
│        ├─ FV_results_26/              # 2026 parliamentary election results (10 constituencies)
│        └─ FV_26_Thomas_Christian.xlsx # 2026 candidate VAA responses (place raw data here)
│
├─ experiments/                         # auto-generated output per run
│  └─ simulations_<N>_alpha_<min>-<max>_noise_<min>-<max>/
│        ├─ parameters.txt
│        ├─ tables/
│        ├─ party_distribution_plots/
│        ├─ voter_travel_plots/
│        ├─ switch_frequency_plots/
│        ├─ all_parties_alpha_comparison/
│        ├─ per_party_alpha_comparison/
│        ├─ candidate_match_bar_plots/
│        ├─ pca_candidate_plots/        # only when simulations=1
│        └─ pca_voter_plots/            # only when simulations=1
│
├─ scripts/
│  ├─ main.py               # entry point — MasterFunction orchestrates the full pipeline
│  ├─ mapping.py            # response-scale mapping (alpha parameterisation)
│  ├─ sampling.py           # proxy voter sampling strategies
│  ├─ noise.py              # noise injection into proxy voter responses
│  ├─ distance_calculation.py  # candidate–voter distance and top-N matching
│  ├─ match_change.py       # voter travel / match-switch analysis
│  ├─ pca.py                # PCA analysis and plots
│  ├─ plots.py              # all visualisations
│  └─ utils.py              # shared helpers (data loading, CI computation, etc.)
│
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## Abstract
Voting Advice Applications (VAA) have in recent years made their entry into European democracies. They are tools deigned to help voters get an overview of listed parties and candidates leading up to election day. Following their rise in popularity, the design of these systems becomes increasingly important as their role in democracies becomes bigger. We investigate a specific part of the design of a Danish VAA, the *Kandidattest* made by *Altinget*, and question the choice of equidistance between response options in a Likert-scale using data from the 2026 Danish general election. Here we show that when the distance between respectively "Agree" and "Strongly agree" and "Disagree" and "Strongly disagree" doubles from a value of 1 to a value of 2, a significant increase or decrease in the fraction of matches is seen for 11 out of the 12 listed political parties. Seen from a user perspective with the same distance change, almost a quarter see a change in advice on party level and more than half see a change on candidate level. These results indicate a lack of robustness in the design of the *Kandidattest*, especially for users of the application, and potential implications for the political parties. They highlight the responsibility for hosts of VAA's and the need for transparency when informing the users about the design of the VAA.

---

## Get data
To obtain the raw data, contact the authors or Altinget.
