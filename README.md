# Updated TBIP Repo

The TBIP pipeline is now simplified into 4 steps. Preparing/Processing, Initializing the Model, Training, and Analyzing the data.

## Running `prepare.py`

By default, this runs the full **prepare pipeline for House speeches**.

### Basic usage
```bash
python -m prepare.prepare \
  --scraped-root scraped-directory \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv