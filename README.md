⚽ Football Analysis | Data Science & Machine Learning

A comprehensive football analytics repository focused on exploring football data, identifying performance patterns, applying statistical techniques, and developing machine learning models to understand and predict football match outcomes.

This project demonstrates the practical application of Python, Data Analytics, Statistics, Data Visualization, Feature Engineering, and Machine Learning to football.

"Football Analysis" (https://github.com/Than-yel/football_analysis/assets/125179701/1d5970a9-ea25-40d8-a555-40284002d5fc)

1. Project Overview

Football generates substantial amounts of data across matches, teams, players, and competitions. Analyzing these records provides opportunities to understand team performance, uncover historical trends, evaluate competitive differences, and investigate factors associated with match results.

The Football Analysis repository explores how statistical analysis and machine learning can transform historical football records into meaningful insights.

The repository includes exploratory football analysis materials and a dedicated English Premier League match-outcome prediction study.

The project combines two major areas of data science:

Descriptive and Exploratory Analytics

Understanding what has happened in football through historical data, statistical summaries, distribution analysis, and visualizations.

Predictive Analytics

Using historical observations to engineer meaningful predictors, train machine learning algorithms, and evaluate their ability to estimate future match outcomes.

The broader objective is to develop a systematic, evidence-based approach to football analytics rather than relying exclusively on subjective opinions or intuition.

---

2. Project Objectives

The primary objectives of this repository are to:

1. Collect, organize, and examine historical football datasets.
2. Perform exploratory data analysis to understand the underlying structure and characteristics of football data.
3. Apply descriptive statistics to quantify team performance and match trends.
4. Investigate relationships between football statistics and match outcomes.
5. Develop meaningful features representing team form, historical performance, and match context.
6. Build machine learning models for football match-outcome prediction.
7. Compare models against simple statistical baselines.
8. Evaluate predictive performance using appropriate classification metrics.
9. Identify methodological limitations, including data leakage, class imbalance, and temporal changes in team performance.
10. Improve the reproducibility and reliability of football analytics workflows.

---

3. Technologies and Tools

Category| Technologies
Programming| Python
Data Manipulation| Pandas, NumPy
Data Visualization| Matplotlib, Seaborn
Statistical Analysis| Descriptive Statistics, Exploratory Data Analysis
Machine Learning| Scikit-learn, XGBoost
Model Algorithms| Logistic Regression, Random Forest, Gradient-Boosted Trees
Development Environment| Jupyter Notebook
Model Serialization| Joblib
Version Control| Git, GitHub

The tools support the full analytical workflow, from preparing raw football records to investigating model predictions and evaluating results.

---

4. Repository Structure

The repository contains exploratory football analysis work alongside a more structured Premier League machine learning study.

Key files and directories include:

football_analysis/
│
├── EDA/
│   └── Exploratory football analysis materials
│
├── epl_match_outcome_26_27/
│   │
│   ├── data/
│   │   └── EPL_data.csv
│   │
│   ├── notebooks/
│   │   └── epl_match_outcome_prediction.ipynb
│   │
│   ├── src/
│   │   └── epl_feature_builder.py
│   │
│   ├── predictions/
│   │   └── matchweek8_champion_predictions.csv
│   │
│   ├── verify_study.py
│   ├── requirements.txt
│   └── README.md
│
├── requirements.txt
└── README.md

This structure highlights the principal analytical and machine learning files rather than every file in the repository.

Main Project Components

Exploratory Data Analysis (EDA)

Contains football analysis materials for investigating datasets and understanding football-related statistical patterns.

Premier League Match Outcome Prediction

A machine learning study that uses historical Premier League match records to predict home wins, draws, and away wins.

Feature Engineering

Contains functionality for generating pre-match variables from historical team performance and maintaining the historical state needed for prediction.

Model Evaluation and Verification

Supports evaluation of the classification models, reproducibility checks, and examination of model limitations.

---

5. Football Data Analysis

Exploratory Data Analysis (EDA) is an important component of this repository.

Before applying machine learning algorithms, football data must be examined to understand its statistical characteristics, quality, limitations, and potential predictive relationships.

5.1 Data Understanding

The initial analytical stage involves examining:

- Dataset dimensions and column types.
- Available football performance indicators.
- Missing values and duplicate observations.
- Temporal coverage and season boundaries.
- Team participation across seasons.
- Consistency of team names and match records.
- Distribution of match outcomes.
- Differences between home and away performances.

These investigations help establish whether the dataset is suitable for meaningful statistical analysis and predictive modeling.

5.2 Descriptive Statistics

Descriptive statistics provide a quantitative summary of football data.

Important statistical techniques include:

Statistical Method| Application
Mean| Average goals, points, shots, or other match statistics
Median| Typical performance less sensitive to extreme observations
Mode| Most frequently occurring values or categories
Variance| Variability in team or match performance
Standard Deviation| Dispersion of performance metrics around their means
Frequency Distribution| Distribution of wins, draws, and losses
Percentiles| Relative positioning of teams or match statistics
Correlation| Relationships between numerical football indicators

These calculations help identify patterns that might otherwise be difficult to observe through individual match records.

5.3 Football Performance Analysis

Relevant analytical questions include:

- How frequently do teams win at home compared with away?
- How does recent team performance compare with overall season performance?
- Which teams consistently outperform their recent historical averages?
- How do goals scored and goals conceded vary across teams?
- How does team performance change during a season?
- How do promoted or returning teams compare with established Premier League clubs?
- Are certain performance indicators associated with a greater probability of winning?

These questions help guide further feature engineering and modeling decisions.

5.4 Data Visualization

Visual analysis complements statistical calculations by making patterns easier to interpret.

Relevant visualization techniques include:

Bar Charts: Comparing teams, match-result frequencies, and performance metrics.

Histograms: Understanding the distribution of numerical statistics such as goals, shots, and points.

Box Plots: Comparing distributions and identifying potentially unusual observations.

Line Charts: Tracking performance changes across matches and seasons.

Scatter Plots: Investigating relationships between numerical variables.

Correlation Heatmaps: Examining relationships between statistical features and identifying possible redundancy.

Visualizations should support meaningful analytical questions rather than simply display available data.

---

6. Premier League Match Outcome Prediction

A dedicated component of this repository investigates the use of historical Premier League data to predict football match outcomes.

Project directory: "epl_match_outcome_26_27/"

6.1 Problem Definition

The objective is to estimate the outcome of a Premier League fixture using information available before the match begins.

The machine learning problem is formulated as a multiclass classification task, with three possible outcomes:

Outcome| Description
H| Home Team Win
D| Draw
A| Away Team Win

Given pre-match information for two competing teams, the model estimates the probability of each result.

For example, a hypothetical prediction might be:

Result| Probability
Home Win| 50%
Draw| 28%
Away Win| 22%

These probabilities represent the model's estimated outcome distribution, not guaranteed results.

6.2 Dataset Description

The match prediction study uses historical English Premier League records covering:

- 2022/2023 season
- 2023/2024 season
- 2024/2025 season
- 2025/2026 season
- 2026/2027 season (partial)

The study dataset contains:

Property| Value
Total Matches| 1,590
Dataset Columns| 26
Teams Represented| 27
Completed Historical Seasons| 4
2026/27 Matches in Final Evaluation| 70
Historical Data Cutoff| October 19, 2026
Dataset File| "EPL_data.csv"

The study's data validation found no missing values or duplicate season fixtures.

The dataset includes match identifiers, team information, results, and match statistics used for historical analysis and feature construction.

The 2026/27 season serves as the final out-of-sample evaluation period for the study.

6.3 Feature Engineering

Football match prediction requires more than simply supplying raw match statistics to a machine learning algorithm.

Features must represent information that would realistically have been available before the fixture.

The study constructs 93 model features, including historical summaries and differences between competing teams.

Important feature categories include:

Recent Team Form

Rolling performance measures from recent matches, including five-match and ten-match historical windows.

Examples include:

- Average points earned.
- Average goals scored.
- Average goals conceded.
- Average goal difference.
- Recent win and draw rates.
- Recent defensive performance.

Home and Away Performance

Measures describing how a team has performed in the relevant playing environment.

A team's home performance may differ significantly from its away performance.

Season-to-Date Statistics

Cumulative performance measurements calculated only from matches already completed in the relevant season.

Historical Team Strength

Elo-based strength estimates that evolve as match results become available.

Fixture Context

Additional information such as match timing, rest days, and historical match counts.

Cold-Start Indicators

Features identifying teams with limited historical information, particularly at the beginning of a season or after promotion.

6.4 Preventing Data Leakage

One of the most important methodological considerations in football prediction is data leakage.

Data leakage occurs when information unavailable at prediction time is introduced into the model's inputs.

For example, using a team's shots on target from the match being predicted would produce an unrealistic forecasting system.

To reduce this risk, the feature engineering process generates match features from previously completed fixtures.

Statistics from a target fixture must not be incorporated into that fixture's own predictors.

The analysis therefore separates:

- Historical information available before kickoff.
- Match information revealed after kickoff or at full time.

This distinction is essential for meaningful prediction.

---

7. Machine Learning Methodology

The prediction study compares three supervised machine learning algorithms.

7.1 Multinomial Logistic Regression

Logistic Regression provides an interpretable linear classification benchmark for estimating class probabilities.

It serves as a useful reference for determining whether more complex models improve performance.

7.2 Random Forest Classifier

Random Forest combines multiple decision trees to capture potentially nonlinear relationships between football performance indicators and match outcomes.

Its ensemble structure can model feature interactions that a simple linear model may miss.

7.3 XGBoost Classifier

XGBoost is a gradient-boosted decision-tree algorithm designed for predictive modeling on structured datasets.

It builds trees sequentially, allowing successive learners to improve the ensemble's predictive performance.

7.4 Model Comparison

The models are compared using chronological validation rather than relying solely on a random train-test split.

The study uses expanding-season validation across the 2023/24, 2024/25, and 2025/26 seasons for model selection, followed by final evaluation on the available 2026/27 matches.

This design better reflects the real-world requirement to predict future fixtures using historical observations.

---

8. Model Evaluation

The performance of a football prediction model must be evaluated using metrics that reflect its actual purpose.

Two important metrics used in this project are:

Classification Accuracy

Accuracy measures the proportion of fixtures for which the model's predicted class matches the recorded result.

Accuracy = Correct Predictions / Total Predictions

Although useful, accuracy does not reveal how confident a model was or whether its predicted probabilities were meaningful.

Multiclass Log Loss

Log loss evaluates the probability assigned to the actual match outcome.

A model is penalized more heavily when it assigns very low probability to an outcome that subsequently occurs.

This is particularly relevant when the objective involves generating probabilities for all three possible match results.

Lower log loss indicates better performance.

8.1 Experimental Results

The recorded final evaluation produced the following results:

Model| Test Accuracy| Test Log Loss
Class-Prior Baseline| 40.0%| 1.095
Logistic Regression| 44.3%| 1.151
Random Forest| 44.3%| 1.022
XGBoost| 45.7%| 1.028

The XGBoost values above represent the original saved notebook run.

Random Forest was selected as the study's preferred model based on average chronological-validation log loss rather than final test accuracy.

An independent verification run reproduced the Random Forest results to the reported precision. XGBoost produced slightly different results under a different dependency environment: 44.3% accuracy and 1.023 log loss.

8.2 Interpretation

The results demonstrate several important considerations.

First, the machine learning models showed improvements over the baseline in accuracy, although the magnitude of improvement was limited.

Second, predictive accuracy alone was insufficient for selecting the preferred model.

Random Forest produced the best recorded final log loss in the original run despite XGBoost having higher recorded accuracy.

Third, the models still struggled with certain outcomes. In particular, the selected Random Forest model recorded 0% recall for draws on the final evaluation set.

The final test dataset also contained only 70 matches, meaning performance estimates are subject to substantial sampling uncertainty.

These results should be interpreted as experimental findings rather than evidence of a consistently reliable football forecasting system.

---

9. Match Forecasting

The project includes a historical forecasting exercise for Premier League Matchweek 8 of the 2026/27 season.

The predictions file is located at:

"epl_match_outcome_26_27/predictions/matchweek8_champion_predictions.csv"

It contains nine Saturday and Sunday fixture forecasts for October 24–25, 2026.

Each forecast provides estimated probabilities for:

- Home Win
- Draw
- Away Win

The models were constructed using completed match information through October 19, 2026.

The study also distinguishes retrospective predictions from genuine forward-looking forecasts. For example, the October 23 Ipswich Town versus Nottingham Forest fixture is explicitly identified as a backcast.

These forecasts are retained as historical research outputs and should not be interpreted as continuously updated live predictions.

---

10. Installation and Setup

Prerequisites

Ensure that Python, Git, and a suitable development environment are installed.

Jupyter Notebook is used for executing and exploring the analytical workflow.

Step 1: Clone the Repository

git clone https://github.com/Than-yel/football_analysis.git

Step 2: Navigate to the Project

cd football_analysis

Step 3: Open the Machine Learning Study

cd epl_match_outcome_26_27

Step 4: Create a Virtual Environment

python -m venv .venv

Activate the environment using the command appropriate for your operating system.

Windows:

.venv\Scripts\activate

Linux/macOS:

source .venv/bin/activate

Step 5: Install Dependencies

pip install -r requirements.txt

The machine learning subproject has its own dependency specification. It should be used when reproducing that study rather than assuming the older repository-level requirements are compatible.

Step 6: Launch Jupyter Notebook

jupyter notebook notebooks/epl_match_outcome_prediction.ipynb

Execute the notebook cells in sequence to inspect the analytical workflow, model training, evaluation, and forecasting outputs.

---

11. Reproducibility and Verification

Reproducibility is an important consideration in machine learning projects.

The repository includes a verification script:

"epl_match_outcome_26_27/verify_study.py"

The independent verification process examined:

- Historical data cutoff and target definitions.
- Chronological training and testing separation.
- Model feature construction.
- Random Forest evaluation metrics.
- Forecast probability outputs.
- Model serialization and reload consistency.

The study also records discrepancies observed between different dependency environments.

Saved notebook outputs should be distinguished from newly reproduced experimental results.

While the project includes reproducibility checks, the original source match records were not independently verified against official football records during the documented verification process.

---

12. Challenges and Limitations

Football prediction is difficult because match outcomes depend on many factors, including player availability, tactical decisions, injuries, team selection, individual errors, and random events.

Several limitations affect this project.

1. Limited evaluation sample

The final 2026/27 evaluation uses only 70 matches, restricting the reliability of conclusions about generalization.

2. Difficulty predicting draws

The selected model recorded zero recall for draws, indicating that its multiclass classification performance is not balanced across outcomes.

3. Promoted teams and historical coverage

Teams entering the Premier League may have little or no relevant top-flight history in the available dataset.

Historical priors and cold-start assumptions can influence early-season predictions.

4. Historical form does not capture every match condition

Rolling averages and Elo estimates summarize team performance but cannot represent every relevant real-world factor.

5. Temporal changes

Team quality, managerial approaches, player personnel, and playing style can change significantly between seasons.

6. Probability reliability

A model may achieve reasonable classification accuracy while producing poorly calibrated probabilities.

7. Computational reproducibility

Differences in package versions or execution environments can lead to variations in model results.

These limitations must be considered when interpreting the project's findings.

---

13. Future Development

Potential improvements to the repository include:

Expanded Exploratory Analysis

Develop more detailed statistical reports on team form, home advantage, goal distributions, seasonal trends, and match-result frequencies.

Advanced Feature Engineering

Investigate additional pre-match predictors such as squad availability, longer-term team strength, promoted-team history, and contextual match information.

Improved Validation

Evaluate the forecasting methodology across additional chronological holdout seasons and use larger out-of-sample evaluation periods.

Probability Calibration

Investigate calibration methods and scoring rules to improve the reliability of predicted match probabilities.

Model Explainability

Use feature importance and explainability methods to better understand which predictors influence model decisions.

Interactive Football Dashboard

Develop an interactive application for exploring historical performance, comparing teams, and displaying analytical results.

Model Deployment

Explore exposing a validated prediction model through an API and integrating it into a football analytics application.

Any future deployment would require additional testing, input validation, monitoring, and evidence of reliable out-of-sample performance.

---

14. Key Learning Outcomes

This project provides practical experience in applying data science principles to a real-world sporting domain.

Major learning areas include:

- Working with historical and time-dependent datasets.
- Applying exploratory data analysis and statistical techniques.
- Translating football knowledge into measurable variables.
- Building features from historical observations.
- Preventing target leakage in predictive modeling.
- Training and comparing classification algorithms.
- Establishing meaningful baseline models.
- Applying chronological model validation.
- Evaluating probabilistic predictions.
- Investigating model weaknesses and reproducibility challenges.

The repository illustrates how football analytics can be approached as a structured data science and machine learning problem.

---

15. Data Source and References

Relevant football resources include:

Official Premier League Website

https://www.premierleague.com/

This website provides general Premier League information, fixtures, standings, and competition statistics.

The match prediction study uses historical match records assembled into its project dataset. The official website link should not be interpreted as proof that every record was downloaded directly from the Premier League.

Additional dataset provenance and original download sources should be documented where available.

---

16. Project Status

Status: Active research and development.

The repository contains analytical materials and an experimental Premier League machine learning study.

The predictive models are research prototypes and have not been established as production-ready forecasting systems.

Future work will focus on strengthening statistical analysis, improving predictive evaluation, and developing more comprehensive football analytics applications.

---

17. Author

Daniel Ndukwe

AI Engineer | Machine Learning | Data Analytics

GitHub: "@Than-yel" (https://github.com/Than-yel)

---

Football Analysis is a data science and machine learning project intended for research, experimentation, and continuous technical development. 

