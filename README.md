# Real-World AI-Driven Analytics Solution Using Python

## Predicting 30-Day Hospital Readmission for Patients with Diabetes

This project develops and evaluates a machine-learning workflow for predicting whether a hospital encounter involving a patient with diabetes will result in readmission within 30 days. The analysis uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository and is implemented in Python using Google Colab.

The project covers data exploration and cleaning, exploratory data analysis (EDA), feature preprocessing, initial Random Forest modeling, hyperparameter optimization, model evaluation, feature importance, and ethical considerations.

## Project Objective

The prediction task is formulated as binary classification:

- **1:** Readmitted within 30 days
- **0:** Readmitted after 30 days or not readmitted

A practical application is to support hospital discharge planning by helping care-management teams identify encounters that may warrant additional post-discharge outreach, follow-up scheduling, medication reconciliation, or transitional-care resources. Model predictions are intended for decision support, not automated clinical decision-making.

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008  
**Download:** https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008

Key dataset characteristics used in this analysis:

| Characteristic | Result |
|---|---:|
| Hospital encounters | 101,766 |
| Original variables | 50 |
| Unique patients | 71,518 |
| Duplicate rows | 0 |
| Final modeling predictors | 44 |
| Numerical predictors | 8 |
| Categorical predictors | 36 |
| Training encounters | 81,412 |
| Testing encounters | 20,354 |

The UCI `IDS_mapping.csv` file is parsed programmatically to obtain readable descriptions for admission type, discharge disposition, and admission source.

## Technical Workflow

1. **Data loading and inspection**
   - Upload and extract the UCI dataset in Google Colab
   - Inspect dimensions, data types, and representative records

2. **Data quality and cleaning**
   - Convert `?` missing-value indicators to `NaN`
   - Check duplicate records and feature-level missingness
   - Remove `weight`, `payer_code`, and `medical_specialty`

3. **Target construction**
   - Create a binary target for 30-day readmission
   - Assess class imbalance

4. **Exploratory data analysis**
   - Target distribution and numerical associations
   - Age, race, and gender patterns
   - Prior inpatient, emergency, and outpatient utilization
   - Discharge disposition
   - A1C testing and medication changes
   - Numerical correlation analysis
   - Repeat patient encounters

5. **Feature preparation**
   - Exclude encounter and patient identifiers
   - Remove the original readmission field to prevent target leakage
   - Exclude EDA-only administrative description columns from modeling
   - Use an 80/20 stratified train-test split

6. **Preprocessing pipeline**
   - Median imputation for numerical features
   - Most-frequent imputation for categorical features
   - One-hot encoding with unknown-category handling
   - `ColumnTransformer` and scikit-learn `Pipeline`

7. **Initial Random Forest model**
   - 100 trees
   - `random_state=42`
   - Evaluate accuracy, precision, recall, F1, ROC-AUC, classification report, confusion matrix, and ROC curve

8. **Optimization and evaluation**
   - `RandomizedSearchCV`
   - Five-fold `StratifiedKFold`
   - F1 scoring for model selection
   - Tune number of trees, maximum depth, minimum samples per leaf, and class weighting
   - Compare initial and tuned Random Forest performance

## Model Results

### Initial Random Forest

| Metric | Score |
|---|---:|
| Training Accuracy | 1.0000 |
| Testing Accuracy | 0.8887 |
| Accuracy Gap | 0.1113 |
| Precision | 0.6875 |
| Recall | 0.0048 |
| F1 Score | 0.0096 |
| ROC-AUC | 0.6574 |

The initial model achieved high overall accuracy but detected very few actual 30-day readmissions. The large training-testing accuracy gap also indicated overfitting.

### Tuned Random Forest

Best selected configuration:

- `n_estimators=150`
- `max_depth=16`
- `min_samples_leaf=2`
- `class_weight="balanced"`
- Best cross-validated F1: **0.2728**

| Metric | Initial | Tuned |
|---|---:|---:|
| Training Accuracy | 1.0000 | 0.6904 |
| Testing Accuracy | 0.8887 | 0.6708 |
| Accuracy Gap | 0.1113 | 0.0196 |
| Precision | 0.6875 | 0.1858 |
| Recall | 0.0048 | 0.5764 |
| F1 Score | 0.0096 | 0.2810 |
| ROC-AUC | 0.6574 | 0.6810 |

Optimization substantially improved minority-class recall, F1 score, ROC-AUC, and the training-testing accuracy gap. The trade-off was lower precision and overall testing accuracy, reflecting more false-positive predictions while detecting substantially more actual readmissions.

## Tools and Libraries

- Python
- Google Colab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Key scikit-learn components include `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `RandomForestClassifier`, `StratifiedKFold`, and `RandomizedSearchCV`.

## Running the Notebook

1. Open the `.ipynb` notebook in **Google Colab**.
2. Download the UCI dataset ZIP file from the link above.
3. Run the notebook from the first cell.
4. When prompted, upload the downloaded dataset ZIP file.
5. The notebook extracts `diabetic_data.csv` and `IDS_mapping.csv` automatically.
6. Run all remaining cells in sequence to reproduce preprocessing, EDA, model training, optimization, and evaluation.

For a clean reproducibility check in Colab, use **Runtime > Restart session and run all** after the dataset ZIP is available for upload.

## Repository Contents

A recommended repository structure is:

```text
.
├── README.md
├── Real_World_AI_Driven_Analytics_Project_Part_1.ipynb
└── report/
    └── PhDAI733_Real-World_AI-Driven_Analytics_Project_Part_1.pdf
```

The UCI dataset should be downloaded from its original source rather than redistributed in the repository unless redistribution is explicitly permitted.

## Ethical and Practical Considerations

The dataset includes sensitive demographic and healthcare information. Historical associations involving age, race, and gender may reflect differences in healthcare access, treatment, documentation, or other underlying factors and should not be interpreted as causal effects.

The model should be evaluated for disparities in false-negative and false-positive rates across patient groups. In practical use, higher recall can reduce missed readmissions but may also increase false positives and the use of limited care-management resources. Predictions should therefore supplement clinical judgment rather than replace healthcare professionals or automate clinical decisions.

## Limitations

- The target is substantially imbalanced, with 11.16% of encounters representing 30-day readmission.
- The initial Random Forest substantially overfits the training data and has very low positive-class recall.
- The tuned model improves recall and F1 but reduces precision and overall accuracy.
- Multiple encounters can belong to the same patient. An encounter-level split may therefore place separate encounters from the same patient in both training and testing partitions, which should be considered when interpreting generalization.
- Feature importance represents predictive contribution within the fitted model and does not establish causation.

## References

- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records*. BioMed Research International, 2014, 781670. https://doi.org/10.1155/2014/781670
- UCI Machine Learning Repository. (2014). *Diabetes 130-US Hospitals for Years 1999-2008*. https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
- scikit-learn Developers. (n.d.). *RandomForestClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
