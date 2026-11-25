# PyBullet Gripper — demo_0

This branch demonstrates PR2/cube grasp simulation, pose data collection,
classifier training and evaluation. The PyQt GUI in `main.py` is a separate
mock prototype. The runnable ML workflow is command line based.

- [Experiment report and workflow](docs/ML_EXPERIMENT_REPORT.md)
- [Simulation and pose data contract](ML_DATA_COLLECTION.md)
- [Coursework brief](OOP%20-%20Final%20CW%20Description2026.pdf)

The tested environment uses Python 3.9, PyBullet 3.2.5, NumPy 2.0.2, pandas
2.3.1, scikit-learn 1.6.1 and joblib 1.5.1. Matplotlib 3.9.4 is used for plots.
The optional GUI requires PyQt6. Run commands from this repository directory
with the environment active.

```bash
python -m unittest discover -s tests
python inspect_dataset.py
```

The checked-in `output/` fixtures include the frozen training data, saved
models and key evaluation records. Write new runs to a fresh directory.
