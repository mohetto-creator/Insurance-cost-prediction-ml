# Predicting Medical Insurance Costs with Machine Learning

**Domain:** Healthcare
**Dataset:** Medical Cost Personal Dataset (`insurance.csv`, public on Kaggle): 1,338 people, 7 columns (age, sex, bmi, children, smoker, region, charges). 1,337 rows remain after removing 1 duplicate.

## Problem
A health insurer has to price each customer correctly. If it charges too little for high-risk people it loses money, and if it charges too much for low-risk people they leave. This project uses machine learning to understand what drives medical costs and to help the insurer price and plan better.

| # | Question | ML type | Models |
|---|---|---|---|
| 1 | How much will this person's medical charges be? | Regression | Linear Regression vs Random Forest |
| 2 | Is this person a high-cost customer (top 25%)? | Classification | Logistic Regression vs Decision Tree |
| 3 | What kinds of customers does the insurer have? | Clustering | KMeans (K = 3) |

## What the data showed
- Smokers pay almost 4x more on average (about $32,000 vs $8,400).
- Smokers are only about 20% of customers but cause about half of all charges.
- Smoking plus obesity is the most expensive combination (about $41,600 on average).
- Age and BMI matter next. Sex, region and number of children matter very little.

## Model results (test data)
| Task | Best model | Result |
|---|---|---|
| Regression | Random Forest (max_depth = 5) | R² 0.87, average error about $2,560 (Linear Regression: R² 0.77) |
| Classification | Decision Tree (depth 3) | Accuracy 93%, precision 96%, AUC 0.88 |
| Clustering | KMeans, K = 3 | Low-risk young non-smokers, medium-risk older non-smokers, high-risk smokers |

## Current losses without machine learning
- If the insurer charged everyone the same average price ($13,279), it would under-charge its expensive customers by about $6 million in this dataset alone, while over-charging low-risk customers who may then leave.
- Each smoker costs about $23,600 more than a non-smoker, and without a model this extra risk is not priced in.
- High-cost customers are only discovered when the bills arrive, which is too late for prevention.
- National picture: smoking-related illness cost the US more than $600 billion in 2018, including over $240 billion in healthcare spending. Obesity adds up to $147 billion a year in medical costs, with obese people spending 42% more on medical care.

## Before vs after machine learning
| Area | Before ML | After ML |
|---|---|---|
| Pricing a new customer | One average price for everyone, off by about $9,400 per person | Random Forest prediction, off by about $2,560 per person (about 73% smaller error) |
| Example customer | A 25-year-old smoker and non-smoker pay the same | Model predicts about $17,900 for the smoker vs about $4,300 for the non-smoker |
| Finding high-cost customers | Known only after claims come in | Flagged at sign-up with 93% accuracy, and 96% of flags are correct |
| Customer strategy | One plan for everyone | 3 risk groups, each with its own pricing and programmes |
| Prevention spending | Spread evenly, or none at all | Focused on 274 smokers who cause about 50% of all charges |
| Money lost on under-priced customers | About $6 million in this dataset | Much smaller, because prices follow real risk |

## Benefits of implementing machine learning
- **Fairer premiums:** each person pays closer to their real expected cost.
- **Less money lost:** high-risk customers are priced correctly instead of being subsidised.
- **Keep good customers:** low-risk customers are not over-charged, so they are less likely to leave.
- **Prevention that pays:** quit-smoking and weight programmes go to the group where they save the most.
- **Industry evidence:** insurers that have scaled AI report 3 to 5 percentage points better loss ratios. On a $1 billion book, a 4-point gain is about $40 million of extra profit a year.

## Recommendations
1. Use the regression model to set premiums based on real risk.
2. Use the classifier to flag likely high-cost customers early and offer prevention.
3. Run quit-smoking and weight programmes for the high-risk group, where they bring the biggest savings.

## Files
| File | Purpose |
|---|---|
| Insurance_Cost_ML.ipynb | Full analysis: cleaning, EDA, regression, classification, clustering |
| insurance.csv | Dataset |
| README.md | This file |

**Tools:** Python, pandas, matplotlib, scikit-learn
