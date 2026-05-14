# Updated TBIP Repo

The TBIP pipeline is now simplified into 4 steps. Preparing/Processing, Initializing the Model, Training, and Analyzing the data.

## Running `prepare.py`

By default, this runs the full **prepare pipeline for House speeches**.

### Basic usage

```bash
python -m prepare.prepare \
  --scraped-root data/scraped/congress_118 \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv
```

### Run for Senate

```bash
python -m prepare.prepare \
  --scraped-root data/scraped/congress_118 \
  --output-root data/congress_118 \
  --base-stopwords data/stopwords.txt \
  --procedural-model data/models/procedural/model.nontest.tsv \
  --chamber senate
```

### Run for both chambers

```bash
python -m prepare.prepare \
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

## Running `init_topics.py`

The init pipeline runs:

1. Poisson Factorization (PF)
2. MALLET LDA
3. MALLET scaling using PF runs

This stage initializes the topic model inputs used later by TBIP.

### 1. Create Python 3.8 Conda Environment

The legacy MALLET + gensim stack requires Python 3.8.

#### Create environment

```bash
conda create -n tbip38 python=3.8 -y
```

#### Activate environment

```bash
conda activate tbip38
```

### 2. Install Dependencies

Install dependencies using conda-forge.

```bash
conda install -c conda-forge \
  numpy \
  scipy \
  pandas \
  scikit-learn \
  tqdm \
  gensim=3.8.3 \
  absl-py \
  configargparse \
  -y
```

### 3. Install MALLET

Download MALLET:

```bash
wget https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
```

Unzip:

```bash
unzip mallet-2.0.8.zip
```

Make executable:

```bash
chmod 755 mallet-2.0.8/bin/mallet
```

Example MALLET path:

```text
/path/to/mallet-2.0.8/bin/mallet
```

### 4. Run Full Init Pipeline

From the repository root:

```bash
python -m init.init_topics \
  --data-root data/congress_114 \
  --chamber senate \
  --mallet-path /path/to/mallet-2.0.8/bin/mallet \
  --num-topics 50 \
  --max-pf-workers 3
```
