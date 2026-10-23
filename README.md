# Predicting Customer Churn: Logistic Regression vs. Lasso

Which telecom customers are about to leave, and does a lasso-penalised logistic regression predict this better than a plain logistic regression? This project compares the two models with a simple benchmark ("always predict no churn") on the same test data.

**Main result:** on 100 random 50/50 splits, the logistic regression beat the benchmark in 93% of the splits, but only by about 0.6 percentage points in test error (13.5% vs. 14.1%). The lasso did not improve on it (14.0%).

## Data

`mlc_churn` from the R package `modeldata`: 5,000 customers, 14.1% churners, 19 predictors (account length, international and voice-mail plan, call minutes, calls and charges by time of day, number of customer service calls). According to the package documentation, the data are artificial.

## Approach

1. Explore the data; drop `state`, `area_code` and the four `charge` columns (almost perfectly correlated with the call minutes).
2. Benchmark: a model that always predicts "no".
3. Logistic regression (`glm`) on a 50/50 train/test split, evaluated with the test error, confusion matrix, sensitivity and specificity.
4. Lasso logistic regression (`cv.glmnet`, `lambda.1se` from 10-fold cross-validation on the training data).
5. Repeat the comparison on 100 random splits (paired comparison on the same splits).
6. ROC curves and AUC; effect of a lower classification threshold (0.2).

## Results

| Model | Test error (mean of 100 splits) | Test error (one split) |
|---|---|---|
| Benchmark (always "no") | 14.1% | 13.6% |
| Logistic regression | 13.5% | 13.4% |
| Lasso (`lambda.1se`) | 14.0% | 13.5% |

- At a threshold of 0.5 the logistic regression finds only about 20% of the churners (sensitivity), the lasso about 3%.
- AUC on the test set: 0.82 (logistic regression), 0.80 (lasso).
- With a threshold of 0.2 the logistic regression finds 67% of the churners, but the test error rises to 18.6%. Which threshold is best depends on the cost of a retention offer versus a lost customer, which is not in the data.

## Limitations

A single, artificial data set; no external validation; no cost information; only 50/50 splits and a small set of models.

## How to run

Open `churn_classification.Rmd` in RStudio and knit it. Required packages: `modeldata`, `glmnet`, `pROC`, `caret`. All random steps use `set.seed(2)`; the R and package versions are printed at the end of the report (`sessionInfo()`).

## Files

- `churn_classification.Rmd`: code and report
- `churn_classification.html`: rendered report

