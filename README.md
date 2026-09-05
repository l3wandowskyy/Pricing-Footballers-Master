# Football Player Market Value Prediction with Machine Learning

This project explores whether machine learning can be used to estimate the market value of offensive football players based on seasonal performance, biometric information, and the competitive context in which they play.

The analysis combines data from **Understat** and **Transfermarkt**, covers the **2017/18 to 2022/23 seasons**, and evaluates separate prediction models for **strikers, wingers, and midfielders**.

## Project Overview

Football player valuation is influenced by many factors, including age, playing time, sporting performance, league quality, team performance, and the player's tactical role. Because these relationships are often nonlinear and vary by position, the project builds separate regression models for three offensive position groups.

The main objectives are to:

- build machine learning models that estimate player market value,
- compare classical regression, tree-based, boosting, and instance-based algorithms,
- evaluate whether PCA improves the performance of simpler models,
- identify the features that contribute most strongly to player valuation,
- compare how valuation drivers differ between strikers, wingers, and midfielders.

The prediction target is **Market Value in EUR**.

## Data Sources

The dataset is constructed from two football data sources:

- **Understat** - seasonal player performance statistics and advanced metrics such as xG, xA, xGChain, and xGBuildup  
  https://understat.com
- **Transfermarkt** - player profile information and historical market values, accessed through an unofficial API  
  https://www.transfermarkt.com  
  https://transfermarkt-api.fly.dev/docs

The Understat Python package used in the extraction notebook is available at:

https://github.com/amosbastian/understat

## Dataset Scope

The original modeling dataset contains:

- **7,845 observations**
- **30 variables**
- seasons from **2017/18 to 2022/23**
- players from the five major European leagues:
  - Premier League
  - La Liga
  - Bundesliga
  - Serie A
  - Ligue 1

The project focuses only on offensive and midfield positions because the available data does not contain enough detailed defensive and goalkeeper statistics for comparable modeling.

The final position groups are:

| Group | Positions |
|---|---|
| Strikers | CF, SS |
| Wingers | RW, LW |
| Midfielders | CM, LM, RM, AM |

## Data Extraction Pipeline

The `Extract_Data.ipynb` notebook builds the analytical dataset.

### 1. Understat extraction

For every league and season, the notebook downloads player-level statistics including:

- matches,
- minutes played,
- goals,
- assists,
- xG,
- xA,
- shots,
- key passes,
- yellow and red cards,
- non-penalty goals,
- non-penalty xG,
- xGChain,
- xGBuildup.

A manually defined UEFA league ranking is also assigned for each league and season.

### 2. Transfermarkt player matching

Players collected from Understat are matched with Transfermarkt records to obtain:

- Transfermarkt player ID,
- position,
- date of birth,
- height,
- nationality,
- preferred foot.

### 3. Historical market value

The notebook retrieves each player's market value history.

For every season, the preferred valuation date is selected from months close to the end of the football season. The priority order is:

1. June
2. May
3. July
4. August
5. April
6. September

If no valuation exists in these months, the annual average is used as a fallback.

### 4. Team-level context

The dataset is enriched with:

- final league position of the player's team,
- number of goals scored by the team,
- `Goal Contribution Share`.

`Goal Contribution Share` represents the percentage of the team's goals to which the player directly contributed through goals or assists.

## Data Preparation

The main modeling workflow is implemented in `preditict_value_footballer.ipynb`.

### Missing values

The initial dataset contains relatively few missing values.

The main missing-value strategy is:

- rows without `Market Value` are removed,
- rows without `Goal Contribution Share` are removed,
- missing `Height` values are replaced with the mean,
- missing `Foot` values are replaced with the mode.

After removing rows without the target variable, **7,700 observations** remain.

### Activity filtering

Players with very limited participation are excluded using the first quartile of:

- matches played,
- minutes played.

The calculated thresholds are:

- Q1 Matches: **10**
- Q1 Minutes: **286.75**

Only players above both thresholds are retained.

This reduces the dataset from **7,700 to 5,494 observations**.

### Per-90-minute features

Season totals are converted into per-90-minute metrics to make players with different amounts of playing time more comparable.

Examples include:

- Goals per 90 Minutes
- Expected Goals per 90 Minutes
- Assists per 90 Minutes
- Expected Assists per 90 Minutes
- Shots per 90 Minutes
- Key Passes per 90 Minutes
- Non-Penalty Goals per 90 Minutes
- Non-Penalty Expected Goals per 90 Minutes
- xG Chain per 90 Minutes
- xG Buildup per 90 Minutes

The notebook also creates:

```text
Minutes Participation Rate =
Minutes Played / (Team Matches * 90)
```

### Player age

Player age is calculated at the end of each analyzed season using the player's birth date and the season end year.

### Categorical variables

Rare nationalities are grouped into an `Other` category before encoding.

In the analyzed dataset, the final `Other` category represents approximately **12.41%** of observations.

Categorical variables such as nationality and preferred foot are encoded numerically for use in the modeling workflow.

## Exploratory Data Analysis

The exploratory analysis is performed separately for strikers, wingers, and midfielders.

It includes:

- descriptive statistics,
- market value distribution analysis,
- position distribution analysis,
- correlation matrices,
- feature-to-target correlation analysis,
- box plots and outlier detection,
- feature scaling,
- position-specific feature selection.

### Market value distribution

Player market value is strongly right-skewed, with many players concentrated at lower values and a relatively small number of highly valued players.

To reduce skewness, the target is modeled using a natural logarithm transformation and transformed back to the original EUR scale for evaluation.

### Outlier removal

Outliers are removed separately for each position group using the IQR rule:

```text
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

The procedure is applied to performance variables such as goals, xG, assists, xA, shots, key passes, cards, non-penalty metrics, xGChain, and xGBuildup.

After outlier removal, the final datasets contain:

| Position group | Observations |
|---|---:|
| Strikers | 1,223 |
| Wingers | 1,016 |
| Midfielders | 1,629 |

## Feature Scaling and Selection

Two scaling strategies are used:

- **StandardScaler** for performance variables expressed mainly as per-90-minute metrics,
- **MinMaxScaler** for variables with naturally limited ranges, such as age, height, UEFA league ranking, team league position, and card statistics.

Correlation analysis is used to remove variables that are redundant, weakly informative, or strongly collinear with other predictors.

Examples include:

- Goals per 90 Minutes when strongly correlated with Non-Penalty Goals per 90 Minutes,
- Expected Goals per 90 Minutes when strongly correlated with Non-Penalty Expected Goals per 90 Minutes,
- selected low-impact variables such as red cards, height, foot, or nationality depending on the position-specific modeling path.

## Machine Learning Models

Seven regression algorithms are compared:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. XGBoost Regressor
6. CatBoost Regressor
7. K-Nearest Neighbors Regressor

The target is log-transformed inside a `TransformedTargetRegressor`, while final predictions and evaluation metrics are calculated in the original EUR scale.

## Training and Evaluation Strategy

For each position group:

1. the dataset is split into **80% training data and 20% test data**,
2. `random_state=42` is used for reproducibility,
3. candidate models are compared using **5-fold cross-validation** on the training set,
4. the strongest models are selected,
5. hyperparameters are tuned using **GridSearchCV with 3-fold cross-validation**,
6. tuned models are evaluated again,
7. final performance is measured on the independent test set.

The main evaluation metrics are:

- **MSE** - Mean Squared Error,
- **RMSE** - Root Mean Squared Error,
- **R²** - Coefficient of Determination.

RMSE is especially useful in this project because it expresses the prediction error directly in euros.

## PCA Experiments

Principal Component Analysis is tested as an alternative modeling path for:

- Linear Regression,
- Ridge Regression,
- Lasso Regression,
- KNN.

The objective is to determine whether reducing dimensionality, multicollinearity, and noise improves the performance of simpler algorithms.

PCA configurations preserve approximately **95% of the variance**, resulting in roughly:

- 10 components for strikers,
- 10 components for wingers,
- 9 components for midfielders.

In most experiments, PCA does not materially improve prediction quality. The main exception is the winger group, where Linear Regression with PCA achieves the best final test result.

## Final Results

The best test-set model differs by position group.

| Position group | Best model | Test RMSE | Test R² |
|---|---|---:|---:|
| Strikers | CatBoost | €10.78M | 0.5742 |
| Wingers | Linear Regression + PCA | €8.02M | 0.5987 |
| Midfielders | CatBoost | €8.61M | 0.6507 |

### Strikers

CatBoost produces the strongest overall result for strikers.

Important valuation drivers include:

- age,
- minutes participation rate,
- team position in the league,
- UEFA league ranking,
- xGChain,
- non-penalty goals,
- non-penalty expected goals.

The result suggests that striker valuation is influenced not only by scoring output but also by age, playing time, and the competitive environment.

### Wingers

The best final result is obtained by Linear Regression with PCA.

For the interpretable non-PCA linear model, the strongest factors include:

- minutes participation rate,
- team position in the league,
- UEFA league ranking,
- age,
- xGBuildup,
- non-penalty expected goals,
- non-penalty goals.

Compared with strikers, winger valuation appears to depend more strongly on playing time, team context, and broader offensive involvement.

### Midfielders

CatBoost achieves the best performance of all position-specific models, with an R² of approximately **0.65**.

The most important predictors include:

- age,
- team position in the league,
- minutes participation rate,
- xGChain,
- UEFA league ranking.

The results indicate that midfielder valuation is driven more strongly by participation in possession and attacking build-up than by direct goal-scoring output.

## Key Findings

The project leads to several main conclusions:

- Player market value can be modeled with machine learning, but model effectiveness depends strongly on the player's position.
- CatBoost performs best for strikers and midfielders, suggesting that nonlinear relationships are important for these groups.
- Linear Regression with PCA performs best for wingers.
- Age, playing time, team league position, and UEFA league ranking are consistently important across position groups.
- Position-specific football metrics matter differently depending on the player's tactical role.
- PCA generally provides limited benefits and does not consistently improve model performance.
- Lasso Regression and KNN are among the weakest approaches across the analyzed position groups.
- Prediction errors remain substantial, with RMSE values of approximately **€8M to €11M**, compared with an average player value of roughly **€15M**.

## Limitations

The results should be interpreted in the context of several limitations:

- market value is an estimated value rather than an actual transfer fee,
- only offensive players and midfielders are included,
- the final position-specific samples are relatively limited,
- several potentially important market factors are unavailable,
- contract length is not included,
- club reputation is only partially represented by sporting results,
- off-field and negotiation-related factors are not modeled,
- injuries, salary information, and contract circumstances are not included.

These omitted factors may explain part of the remaining prediction error.

## Future Improvements

Potential extensions of the project include:

- adding contract duration and contract expiry information,
- developing a stronger measure of club reputation,
- adding injury and availability history,
- incorporating salary and transfer-related information,
- expanding the dataset with additional seasons and leagues,
- developing dedicated models for defenders and goalkeepers using position-specific statistics,
- testing additional ensemble and time-aware modeling approaches.

## Repository Notebooks

### `Extract_Data.ipynb`

Responsible for:

- downloading Understat statistics,
- retrieving Transfermarkt player profiles,
- retrieving historical market values,
- merging player and season information,
- filtering offensive positions,
- enriching the dataset with team league position and goal contribution.

### `preditict_value_footballer.ipynb`

Responsible for:

- data cleaning,
- missing-value handling,
- feature engineering,
- per-90-minute transformations,
- exploratory data analysis,
- outlier removal,
- feature scaling and selection,
- position grouping,
- model training,
- cross-validation,
- hyperparameter tuning,
- PCA experiments,
- test-set evaluation,
- feature-importance analysis.

## Technology Stack

The project is implemented in Python and Jupyter/Google Colab.

Main libraries:

```text
pandas
numpy
requests
aiohttp
nest_asyncio
understat
openpyxl
matplotlib
seaborn
scikit-learn
xgboost
catboost
```

## Installation

A basic local environment can be prepared with:

```bash
pip install pandas numpy requests aiohttp nest_asyncio understat openpyxl \
    matplotlib seaborn scikit-learn xgboost catboost
```

The extraction notebook also uses the unofficial Transfermarkt API implementation referenced in the notebook.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/l3wandowskyy/Pricing-Footballers-Master.git
cd Pricing-Footballers-Master
```

### 2. Run the extraction notebook

Open and execute:

```text
Extract_Data.ipynb
```

The notebook generates intermediate Excel files and ultimately prepares the player dataset used for analysis.

A league-position input file is also used to enrich the player data with final team position and team goals.

### 3. Run the modeling notebook

Open and execute:

```text
preditict_value_footballer.ipynb
```

This notebook performs preprocessing, exploratory analysis, position-specific model training, tuning, PCA experiments, final evaluation, and feature-importance analysis.

## Reproducibility

Where applicable, the modeling workflow uses:

```python
random_state = 42
```

The same train/test split and cross-validation setup can therefore be reproduced when the input dataset and library versions remain unchanged.

## Disclaimer

This project was developed for academic and analytical purposes.

Transfermarkt market values are estimates and should not be interpreted as guaranteed transfer prices. The project is not affiliated with Understat or Transfermarkt.