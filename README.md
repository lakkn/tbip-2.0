# Updated TBIP Repo

The TBIP pipeline is now simplified into 4 steps. Preparing/Processing, Initializing the Model, Training, and Analyzing the data.

## Running `prepare.py`

By default, this runs the full **prepare pipeline for House speeches**.

### Basic usage

```bash
python -m congress_pipeline.preprocess.prepare \
  --scraped-root data/scraped/congress_118 \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv
```

### Run for Senate

```bash
python -m congress_pipeline.preprocess.prepare \
  --scraped-root data/scraped/congress_118 \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv \
  --chamber senate
```

### Run for both chambers

```bash
python -m congress_pipeline.preprocess.prepare \
  --scraped-root data/scraped/congress_118 \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv \
  --chamber both
```

### Helpful optional flags

- `--json-limit 1000` to test on a smaller subset of scraped JSON files
- `--split-chunksize 50000` for large CSV splitting
- `--verbose` for detailed logging
