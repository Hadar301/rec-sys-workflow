# rec-sys-workflow
this repo will hold the workflows in rec-sys TODO add link for VP validated patterns

## Batch Recommendation Pipeline - train-workflow.py
do the training for batch recommendation system rec-sys VP


### Pipeline Flow
1. Load data from Feast feature store
2. Train the two-tower model
3. Generate and push recommendations to online store

## Code Quality

This repository enforces code quality standards using:

- **flake8**: Python linting and style checking (100 character line limit)
- **isort**: Import sorting and organization

The pre-push hook will automatically run `isort` and `flake8` checks and show warnings for any quality issues, but won't block pushes.

### Manual Quality Checks

First install flake8 by 
```bash
pip install flake8-pyproject
```

To check and fix code quality issues locally:

```bash
# Check for flake8 errors
flake8 .

# Check import sorting
isort --check-only --diff .

# Fix import sorting automatically
isort .
```
