# UP Classifier Agent

UP Classifier Agent is a Flask web application for Bilibili creator profiling and business value analysis. Users can search by UID or creator name, open a dashboard, and view model predictions, value scores, feature comparisons, similar creators, and diagnostic suggestions.

## What It Does

- Crawls recent Bilibili creator data by UID.
- Builds a 10-dimensional feature profile from plays, danmaku, comments, video length, upload frequency, and comment repetition.
- Runs a local scikit-learn classifier saved in `classifier/up_classifier_10dim.pkl`.
- Computes confidence, feature contributions, a normalized business value score, and score percentile buckets.
- Shows a bilingual web dashboard with Chart.js visualizations and recommendation text.
- Optionally calls DeepSeek for a natural-language diagnostic summary when `DEEPSEEK_API_KEY` is configured.

## Project Structure

```text
.
├── app_1.py                         # Flask app, crawler, scoring APIs, dashboard routes
├── analysis.py                       # Shared feature column definition
├── requirements.txt                  # Python dependencies
├── classifier/
│   └── up_classifier_10dim.pkl       # Trained local classifier
├── database/
│   └── upfile_data_labeled_10.csv    # Feature table used by the app
├── templates/
│   ├── home.html                     # Search page
│   └── dashboard.html                # Dashboard shell
├── static/
│   ├── dashboard.css                 # Dashboard styling
│   └── dashboard.js                  # Dashboard client logic and charts
├── ENGINEERING_OVERVIEW.md           # Existing engineering summary
└── PROJECT_DESIGN.md                 # System design notes
```

## Core Features

The model uses these 10 features:

- `avg_comment_scraped`
- `avg_danmaku`
- `avg_length`
- `avg_play`
- `comment_repetition`
- `danmaku_missing_rate`
- `med_danmaku`
- `med_play`
- `std_length`
- `upload_freq`

The dashboard groups them into interaction, play statistics, interaction behavior, and video length views.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional `.env`:

```text
DEEPSEEK_API_KEY=your_api_key
```

## Run

```bash
python app_1.py
```

Then open the local Flask URL printed by the app, usually `http://127.0.0.1:5000`.

## Main API Routes

- `GET /` - search page
- `GET /dashboard?uid=<uid>` - dashboard page
- `GET /api/search?q=<query>` - search by UID or creator name
- `GET /api/recommendations` - random creator recommendations
- `GET /api/predict/<uid>` - crawl/update data and return prediction
- `GET /api/stats/good` - benchmark statistics for high-value creators
- `GET /api/prescription/<uid>` - feature contribution explanation and suggestions
- `GET /api/peers/<uid>` - similar creators based on z-score cosine similarity

## Notes

- Bilibili crawler behavior depends on upstream API availability, cookies, WBI signing, and rate limits.
- New UID prediction can update `database/upfile_data_labeled_10.csv`.
- The app can fall back to existing CSV data when a crawl fails for a UID already present in the dataset.
