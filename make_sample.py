"""Create a small sample (household + individual CSV) from the training data to try the site.

Usage: python make_sample.py --data-dir path/to/csvs --country C --n 300 [--no-labels]
Writes sample_<country>_hhold.csv and sample_<country>_indiv.csv in the current folder.
Note: this samples the training data, so scores will look better than on truly unseen households.
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='.'); ap.add_argument('--country', default='C')
ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--no-labels', action='store_true', help="drop the 'poor' column")
a = ap.parse_args()
hh = pd.read_csv(os.path.join(a.data_dir, f'{a.country}_hhold_train.csv'))
iv = pd.read_csv(os.path.join(a.data_dir, f'{a.country}_indiv_train.csv'))
s = hh.sample(a.n, random_state=a.seed)
iv = iv[iv['id'].isin(s['id'])]
if a.no_labels:
    s = s.drop(columns=['poor'])
s.to_csv(f'sample_{a.country}_hhold.csv', index=False); iv.to_csv(f'sample_{a.country}_indiv.csv', index=False)
print(f'wrote {len(s)} households and {len(iv)} individuals')
