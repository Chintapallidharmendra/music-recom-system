
## Live Simulator
python -m service.live_simulator     --scenario normal     --n 5000     --interval 0.05

## Observing Live Feedback
tail -f mlruns/live_feedback.jsonl

## Introducing Drift Through Live Simulator
python -m service.live_simulator     --scenario preference_shift   --n 5000 --start-after 500 --magnitude 0.8    --interval 0.05

python -m mlops.drift_report     --live mlruns/live_feedback.jsonl