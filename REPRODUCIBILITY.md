# Reproducibility

## 1. Environment

The project was verified using:

- Python 3.11
- numpy 2.4.3
- pandas 2.3.3
- matplotlib 3.11.1
- scikit-learn 1.9.0

The exact core modelling dependencies are pinned in `requirements.txt`.

Jupyter front-end/kernel packages are not pinned as core dependencies. The `.ipynb` notebooks can be opened with any compatible notebook environment. On the verification machine, Windows Application Control blocked the ZeroMQ DLL used by the Jupyter kernel, so the notebook code cells were additionally verified by sequential Python execution as documented below.

From the repository root on Windows:

    py -3.11 -m venv .venv
    source .venv/Scripts/activate
    python -m pip install -r requirements.txt

The project uses classical machine-learning models and does not require a GPU.

---

## 2. Raw Data

The project uses the Hotel Booking Demand dataset.

Dataset page:

    https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Download the Hotel Booking Demand archive through the Kaggle web interface, extract `hotel_bookings.csv`, and place the unmodified file at:

    data/raw/hotel_bookings.csv

A Kaggle account/sign-in may be required for browser download.

The raw file is treated as immutable and is not modified in place.

Verified raw-file fingerprint:

    SHA-256: d10cdeb07de9309007f528d6499a2c0467ae7a6b287b80ffd759e11e61801639

Verified raw dimensions:

    119,390 rows × 32 columns

The hash allows an independent reproduction attempt to confirm that it is using the exact raw CSV version used for the reported results.

Notebook 03 generates:

    data/processed/hotel_bookings_cleaned.csv

The processed CSV is reproducible from the raw data and is intentionally excluded from Git.

---

## 3. Random Seeds

The modelling workflow uses:

    RANDOM_STATE = 42

The fixed seed is used for:

- reproducible sampling where applicable
- stratified train/test splitting
- shuffled stratified cross-validation
- Decision Tree
- Random Forest
- Histogram Gradient Boosting
- other stochastic model components where supported

Notebook 01 also uses:

    random_state=42

for its reproducible sample display.

---

## 4. Train/Test Protocol

The modelling workflow uses:

- 80% training data
- 20% held-out test data
- stratification by `is_canceled`
- random state 42

Final split:

- training rows: 95,367
- held-out test rows: 23,842

The held-out test set is not used for model or threshold selection.

---

## 5. Cross-Validation

Model comparison uses 5-fold stratified cross-validation on the training data.

Decision Tree hyperparameters are selected using training-only cross-validation with ROC-AUC as the tuning metric.

The cost-sensitive threshold is also selected without tuning on the held-out test set.

For threshold selection, the final Random Forest generates 5-fold out-of-fold probabilities on the training data. Thresholds from 0.01 through 0.99 are evaluated using those OOF probabilities.

The selected threshold is then fixed before final held-out evaluation.

---

## 6. Leakage Policy

Two direct outcome-leakage columns are removed during cleaning:

- `reservation_status`
- `reservation_status_date`

Three additional variables are excluded from the booking-time predictive feature set:

- `assigned_room_type`
- `booking_changes`
- `days_in_waiting_list`

These additional variables may contain information that develops after the original reservation.

Preprocessing required for model fitting is contained inside scikit-learn pipelines so fitted transformations are learned from training data.

Rare country categories below 1% training frequency are grouped into `Other` inside the modelling pipeline.

The reusable booking-time transformation, split definition, cross-validation constructor, preprocessing policy, and canonical Random Forest definition are centralised in `src/project_utils.py`. Notebooks 05 through 08 import this shared implementation rather than maintaining independent copies.

---

## 7. Notebook Execution Order

From the repository root:

    cd notebooks

Run the notebooks in this exact order:

1. `01_data_understanding.ipynb`
2. `02_data_quality_leakage.ipynb`
3. `03_data_cleaning.ipynb`
4. `04_exploratory_data_analysis.ipynb`
5. `05_feature_engineering_preprocessing.ipynb`
6. `06_model_training.ipynb`
7. `07_model_evaluation.ipynb`
8. `08_cost_sensitive_threshold.ipynb`

Notebook 03 must run before Notebooks 04 through 08 because it creates the canonical processed dataset they consume.

---

## 8. Expected Outputs

Notebook 03 generates:

    data/processed/hotel_bookings_cleaned.csv

Notebook 07 generates:

    results/metrics.csv
    results/confusion_counts.csv
    results/rf_logistic_feature_comparison.csv

and final evaluation figures under:

    results/figures/

Notebook 08 generates:

    results/threshold_sweep.csv
    results/decision_summary.csv

and the threshold-decision figures:

    results/figures/threshold_cost_curve.png
    results/figures/threshold_precision_recall_tradeoff.png
    results/figures/recommended_threshold_confusion.png

Expected final model ranking places Random Forest first on the held-out test set.

Verified Random Forest held-out results:

- Accuracy at explicit probability threshold 0.50: 0.8925
- Precision at threshold 0.50: 0.8815
- Recall at threshold 0.50: 0.8205
- F1 at threshold 0.50: 0.8499
- ROC-AUC: 0.9584
- Average Precision: 0.9406

---

## 9. Cost-Sensitive Decision

The threshold analysis uses the explicit relative-cost assumption:

    False Positive cost = 2
    False Negative cost = 1

Using training-only 5-fold OOF probabilities, the minimum-cost threshold is:

    0.61

On the held-out test set:

- expected cost at threshold 0.50: 3,537 relative units
- expected cost at threshold 0.61: 3,291 relative units
- reduction: 246 relative units
- percentage reduction: 6.96%

This threshold is conditional on the stated cost assumption and is not claimed to be a universal hotel-industry threshold.

---

## 10. Runtime and Hardware

The workflow was verified on a Windows laptop using CPU execution only. No GPU is required.

A final clean sequential execution was performed after removing the generated processed dataset and result artifacts. Starting from `data/raw/hotel_bookings.csv`, all eight notebooks executed successfully in numerical order and regenerated the downstream artifacts.

Measured clean-run times on the verification machine:

- Notebook 01: 0.22 minutes
- Notebook 02: 0.01 minutes
- Notebook 03: 0.02 minutes
- Notebook 04: 0.33 minutes
- Notebook 05: 0.03 minutes
- Notebook 06: 1.03 minutes
- Notebook 07: 0.70 minutes
- Notebook 08: 0.41 minutes
- **Total clean 01–08 execution: 2.76 minutes**

Runtime depends on CPU, available memory, background workload, operating-system policy, and library versions. These measurements demonstrate successful execution on the verification machine rather than a performance guarantee.

---

## 11. Manual Steps

1. Clone the repository.
2. Confirm `data/raw/hotel_bookings.csv` is present.
3. Create the Python 3.11 virtual environment.
4. Install `requirements.txt`.
5. Enter the `notebooks` directory.
6. Run Notebooks 01 through 08 in numerical order.

The raw dataset is obtained separately and placed in `data/raw/`; the modelling workflow does not download data programmatically.

---

## 12. Known Environment Note

During development, Jupyter kernel execution through `nbclient` could not be used on the verification machine because Windows Application Control blocked the ZeroMQ DLL.

This was an environment/security-policy issue rather than a notebook-code failure.

To verify the project despite that restriction, notebook code cells were executed sequentially using Python 3.11 in the project virtual environment. The final versions of the modelling notebooks were successfully executed end-to-end using that method.

A standard Jupyter environment without that local Application Control restriction can run the notebooks normally.

---

## 13. Reproducibility Status

The following have been verified:

- raw dataset loads successfully
- cleaning generates the expected processed dataset
- all eight notebooks have valid Python code cells
- leakage exclusions are applied before modelling
- fixed train/test split reproduces the same class proportions
- all five required models execute
- final held-out metrics are generated
- `results/metrics.csv` contains all five models with no missing values
- all held-out confusion matrices contain exactly 23,842 observations
- Notebook 07 and Notebook 08 use consistent explicit probability-threshold semantics
- Notebook 07 and Notebook 08 reproduce the same Random Forest default-threshold decision cost
- final evaluation and threshold figures are persisted under `results/figures/`
- threshold selection uses training-only OOF probabilities
- threshold sweep contains 99 thresholds from 0.01 to 0.99
- `results/threshold_sweep.csv` contains no missing values
- recommended threshold reproduces as 0.61

A final clean sequential 01–08 execution was also completed successfully from the raw dataset after generated processed/result artifacts were removed. Notebook 03 regenerated the canonical processed dataset, and Notebooks 07–08 regenerated the final evaluation and decision artifacts.

The reported results therefore correspond to code that was successfully reproduced from the raw dataset in the documented Python 3.11 environment.
