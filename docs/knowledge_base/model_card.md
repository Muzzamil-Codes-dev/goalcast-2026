# Model Card — GoalCast W/D/L + Goals

- **Intended use**: educational forecasting of international football results and the FIFA
  World Cup 2026. Not for betting.
- **Inputs**: Elo ratings, recent form, rolling goals, rest days, home/host advantage.
- **Outputs**: win/draw/loss probabilities, expected goals, most-likely scoreline,
  tournament-advancement and champion probabilities.
- **Limitations**: no injuries, squad changes, or weather; draws are inherently hard to
  predict; international fixtures are sparse compared with club football.
- **Monitoring**: every prediction is logged and scored against the real result once known
  (realized log loss / Brier), with data-drift checks on the input features.
