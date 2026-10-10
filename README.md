# Customer Retention & Revenue Intelligence Platform

This is an end-to-end churn prediction project. Raw data lives in MySQL, gets cleaned and turned into point-in-time features in Python, and goes through several rounds of model experiments. The result is an XGBoost model behind a Streamlit app that ranks customers by churn risk.

In short, the pipeline works but the model is only modestly better than chance. I've written up both parts. The second part ended up teaching me more than the first.

[Live demo](#) (link to come) · [Data engineering notebook](notebooks/data_loading_profiling_cleaning.ipynb) · [ML notebook](notebooks/machine_learning_pipeline.ipynb)

---
## What I was trying to answer

A retention team can't contact every customer, so they need to know who to look at first. The question I set myself was:

> At the start of a given week, using only what's known about each active customer at that point, can we rank them by how likely they are to churn in the next 30 days?

That makes this a prioritisation problem more than a classification one. A model can output a probability for every customer and still be useless if its ranking isn't better than a simple baseline on future data.

## The data

There are five source tables in MySQL:

- `raw_accounts`: customer identity and signup info
- `raw_subscriptions`: subscription and billing history
- `raw_feature_usage`: product usage
- `raw_support_tickets`: support interactions
- `raw_churn_events`: recorded churn

Each row in the modelling dataset is a **customer-week**, not a customer. For a snapshot date *t*:

- features only use data from on or before *t*
- the label `churn_next_30d` is 1 if a churn event happens in the 30 days after *t*
- only customers who are still eligible at *t* get a row

Scoring happens weekly and the prediction window is 30 days. Those are two separate settings, and I mixed them up a few times early on.

### Avoiding leakage

Leaking the future into the training data is very easy here. For example, you could compute a customer's total usage across their whole history and attach it to a row from six months before they left. To guard against that, snapshots, labels and features are built in separate steps (`SnapshotBuilder`, `TargetBuilder`, `FeatureEngineer`), and I checked the outputs of each before training. I haven't done a formal leakage audit, so I wouldn't claim it's airtight.

## How it's put together

| Part | What I used |
| --- | --- |
| Storage | MySQL locally, Aiven MySQL for the hosted demo |
| Cleaning | My own extraction / profiling / cleaning classes (`db_connector.py`, `data_profiler.py`, `data_cleaner.py`) |
| Snapshots & labels | `SnapshotBuilder`, `TargetBuilder` |
| Features | `FeatureEngineer` |
| Models | scikit-learn, XGBoost, CatBoost, plus my own training/tuning/evaluation modules |
| Scoring | `ChurnPredictor` + saved Joblib artifacts |
| App | Streamlit |

The data engineering notebook covers extraction, profiling and cleaning. Once the schema was clean I moved it to Aiven so the hosted app had something to read from. The app only does inference and never retrains.

## Modelling

The ML notebook follows roughly this order:

1. Build weekly snapshots and 30-day labels
2. Engineer point-in-time features
3. Fit a reusable preprocessor for numeric and categorical columns
4. Split by time: earlier weeks for development, later weeks for testing
5. Baselines: Logistic Regression, Random Forest, XGBoost, CatBoost
6. Tune with temporal cross-validation (I tried different fold setups and gaps)
7. Look at score distributions, thresholds and ranked lists
8. Refit the chosen XGBoost config on the development data
9. Evaluate once on the held-out test period
10. Save the model, preprocessor and feature column order

I didn't use a random split because it would mix the same customers across train and test, with overlapping 30-day windows on top of that. Splitting by time doesn't solve every dependency between repeated snapshots, but it's a lot closer to how the model would actually be used.

## Where it got difficult

Writing the training code was the easy part. The hard part was finding out that more tuning and more complex models didn't really move the needle.

Some early runs predicted zero churners at the default 0.5 threshold. I tried class weights, regularisation, other model families and bigger hyperparameter searches, and none of them made a big difference. I also switched from a 7-day horizon to 30 days. That changes what the model is predicting, but it doesn't by itself make the model more useful.

I focused on **PR-AUC** (with ROC-AUC alongside it) instead of accuracy. Only about 12% of rows are positive, so a model that predicts "nobody churns" gets 88% accuracy while being completely useless. As you'll see below, that's more or less what happened at the 0.5 threshold.

Precision vs recall is the usual trade-off. Flag more people and you catch more churners but waste outreach. Flag fewer and you miss people. The right balance depends on contact cost, customer value and how many people the team can handle. I don't have any of those numbers, so I didn't pick a "correct" threshold.

## Results on the held-out test set

| Metric | Value |
| --- | ---: |
| Customer-week rows | 6,383 |
| Churn-positive rows | 769 |
| Prevalence | 12.05% |
| PR-AUC | 0.1474 |
| PR-AUC ÷ prevalence | 1.22× |
| ROC-AUC | 0.5729 |
| Brier score | 0.1075 |
| Accuracy @ 0.5 | 87.95% |
| False negatives @ 0.5 | 769 |
| False positives @ 0.5 | 0 |

How to read this: the model is a bit better than random, but not by much. PR-AUC of 0.147 against a 0.12 baseline is a small lift, and a ROC-AUC of 0.57 is weak. At a 0.5 cutoff it flagged nobody, so the 88% accuracy is just the class imbalance. I haven't tested calibration, so the scores in the app shouldn't be read as real probabilities.

## Treating it as a weekly ranked list

A fixed threshold didn't make sense here, so I looked at the output the way a team would use it: "we can contact K people this week, so how many future churners are in our top K?"

The test period covers 15 weekly snapshots:

| Weekly capacity | Churners captured | Recall (overall) | Mean precision@K | Mean lift@K | Mean % of eligible customers contacted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top 25 | 59 / 769 | 7.67% | 15.73% | 1.35× | 5.90% |
| Top 50 | 104 / 769 | 13.52% | 13.87% | 1.18× | 11.81% |
| Top 75 | 157 / 769 | 20.42% | 13.96% | 1.18× | 17.71% |
| Top 100 | 235 / 769 | 30.56% | 15.67% | 1.31× | 23.62% |

With 100 contacts a week, the list caught 235 of the 769 churn rows, and weekly precision averaged about 15.7% against a 12% base rate. That's some enrichment, but not a lot.

Some caveats:

- These counts are customer-weeks. 235 rows doesn't mean 235 different people.
- Recall is pooled across all 15 weeks, while precision and lift are averages of the weekly values, so they aren't directly comparable.
- I haven't yet compared this against a random list or a simple business rule (e.g. "most recent support tickets"). That comparison needs to happen before claiming the model beats them.
- Catching a churner isn't the same as saving them. Outreach cost, revenue at risk and whether contact actually changes anything weren't measured.

This part changed how I see the project. Even with a weak classifier, looking at it as a ranking under a capacity limit shows whether it has *any* practical value. Here it has a little.

## The app

The Streamlit app loads the saved model, preprocessor and feature list. It pulls data from the hosted MySQL, builds a snapshot for the chosen date, runs it through the same feature pipeline and ranks the customers.

You can:

- choose a historical prediction date (the 30-day window follows from it)
- view the risk ranking and adjust the threshold
- switch between top 25 / 50 / 75 / 100 capacity
- look up individual customers, see the score distribution, and download the predictions
- see a summary of the model's evaluation

All of this runs on historical data (the default date is 2024-12-01). Opening the app today doesn't score any real current customers.

```text
Aiven MySQL
    ↓
SnapshotBuilder + FeatureEngineer
    ↓
saved preprocessor + feature columns
    ↓
saved XGBoost model
    ↓
ChurnPredictor  →  Streamlit (top-K view)
```

The artifacts are expected in:

```text
models/v2_30d/production/
    xgboost_model.joblib
    preprocessor.joblib
    feature_columns.joblib
```

## Repo layout

```text
customer-retention-intelligence-platform/
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── data_loading_profiling_cleaning.ipynb
│   └── machine_learning_pipeline.ipynb
├── my_src/
│   ├── db_connector.py
│   ├── data_profiler.py
│   ├── data_cleaner.py
│   ├── feature_engineering/
│   │   ├── snapshot_builder.py
│   │   ├── target_builder.py
│   │   └── feature_engineer.py
│   ├── modeling/
│   │   ├── model_evaluator.py
│   │   ├── model_tester.py
│   │   ├── model_trainer.py
│   │   ├── model_tuner.py
│   │   └── preprocessor.py
│   └── prediction/
│       ├── churn_explainer.py
│       └── churn_predictor.py
├── models/
│   └── v2_30d/
│       └── production/
│           ├── xgboost_model.joblib
│           ├── preprocessor.joblib
│           └── feature_columns.joblib
├── certificates/
│   └── ca.pem
├── .gitignore
├── requirements.txt
└── README.md
```

Database settings go in a `.env` file:

```dotenv
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=cleaned_ravenstack_dw
MYSQL_SSL_CA=certificates/ca.pem
```

`MYSQL_SSL_CA` is only needed for the hosted (Aiven) connection. Keep `.env` out of git. The `ca.pem` in the repo is a public CA cert, not a secret. The hosted app also needs the same values set as Streamlit secrets.

The notebooks are a record of how I developed this, not a polished reproduction script. Some of them point to local folders or earlier experiment outputs, so they may not run cleanly from a fresh clone.

## Limitations and what I'd do next

- The model's signal is weak. Better features (behavioural, contractual, customer experience) probably matter more than more tuning.
- Evaluation is per customer-week. Customer-level evaluation would be easier to interpret.
- There's no comparison yet against random or rule-based ranking.
- Scores aren't calibrated.
- There's no evidence that contacting flagged customers actually reduces churn. That would need an A/B test.
- It's a historical demo, with no monitoring or drift checks.

## What I took away from it

Building a working ML pipeline and building a model that's actually useful are two different jobs. I got the first one done properly. The second one was limited by the data, and no amount of tuning was going to change that. I decided to ship it anyway with the numbers shown as they are, because I'd rather someone reviewing this sees exactly what the model can and can't do.

---

**Stack:** Python, pandas, NumPy, SQLAlchemy, MySQL (Aiven), scikit-learn, XGBoost, CatBoost, Joblib, Streamlit

**Jesse Kipchumba**, MSc Data Analytics
