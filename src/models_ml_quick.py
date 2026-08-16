"""Fast classical-ML entrypoint used by the documented rebuild command.

The pooled ML implementation in ``models_ml.py`` already trains the four PRD
classical regressors: Ridge, RandomForest, XGBoost, and SVR.  Keep this file as
a stable, user-facing command name for the quick rebuild path.
"""

from models_ml import main


if __name__ == "__main__":
    main()
