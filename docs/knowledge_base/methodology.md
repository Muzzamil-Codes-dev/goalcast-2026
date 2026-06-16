# GoalCast Methodology

GoalCast predicts match outcomes with two complementary models:

- A **calibrated XGBoost classifier** estimates win/draw/loss probabilities. Isotonic
  calibration is applied so the probabilities are trustworthy rather than overconfident.
- A **Dixon-Coles bivariate Poisson model** estimates expected goals for each team and the
  full scoreline distribution. It also powers the Monte Carlo tournament simulation.

Models are evaluated on a **time-based split** (train on older matches, test on recent ones)
to avoid look-ahead leakage. We grade probabilities with **log loss** and **Brier score**,
and check **calibration curves**, not just accuracy.

The strongest single feature is the **Elo rating difference** between the two teams.
