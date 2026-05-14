import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

def parse_doc_topics(fpath):
    with open(fpath) as infile:
        return np.array([
            [float(x) for x in line.strip().split("\t")[2:]]
            for line in infile
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", help="Directory where all poisson run subdirectories are saved are saved")
    parser.add_argument("--glob_pattern", help="Pattern of the subdirectories poisson runs are saved. Use quotes")
    parser.add_argument("--input_mallet_dir")
    parser.add_argument("--beta_fname", default="beta.npy")
    parser.add_argument("--theta_fname", default="doctopics.txt")
    args = parser.parse_args()

    document_sums = []
    topic_sums = []
    
    print("Collecting saved estimates...")
    for p in tqdm(Path(args.base_dir).glob(args.glob_pattern)):
        document_shape = np.load(p / "document_shape.npy")
        document_rate = np.load(p / "document_rate.npy")
        document_loc = document_shape / document_rate
        document_sums.append(document_loc.sum(-1))

        topic_shape = np.load(p / "topic_shape.npy")
        topic_rate = np.load(p / "topic_rate.npy")
        topic_loc = topic_shape / topic_rate
        topic_sums.append(topic_loc.sum(-1))

    document_means = np.array(document_sums).mean(-1)
    topic_means = np.array(topic_sums).mean(-1)
    
    beta = np.load(Path(args.input_mallet_dir, args.beta_fname))
    theta = parse_doc_topics(Path(args.input_mallet_dir, args.theta_fname))

    bm = topic_means.mean() or beta.shape[1]
    tm = document_means.mean() or theta.shape[1]
    beta = beta * bm
    theta = theta * tm

    np.save(Path(args.input_mallet_dir, "beta_scaled.npy"), beta)
    np.save(Path(args.input_mallet_dir, "theta_scaled.npy"), theta)
    
    np.save(Path(args.input_mallet_dir, "topic_word.npy"), beta[:, :-1] + 1e-6)
    np.save(Path(args.input_mallet_dir, "doc_topic.npy"), theta)