import os
from optparse import OptionParser

import numpy as np

from run_folds import run_folds


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--folds', type=int, default=5,
                      help='Number of test folds: default=%default')
    parser.add_option('--max_seq_length', type=int, default=256,
                      help='max seq length: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--save-steps', type=int, default=94,
                      help='Steps between each save: default=%default')
    parser.add_option('--basedir', type=str, default='splits/',
                      help='Output directory from split_data.py: default=%default')
    parser.add_option('--transformers-dir', type=str, default='../transformers/',
                      help='Transformers dir: default=%default')
    parser.add_option('--n-seeds', type=int, default=5,
                      help='Number of random seeds to try: default=%default')
    parser.add_option('--first-seed', type=int, default=0,
                      help='Seed to start at: default=%default')
    parser.add_option('--last-seed', type=int, default=4,
                      help='Last seed to run: default=%default')
    parser.add_option('--device', type=int, default=None,
                      help='CUDA device: default=%default')
    parser.add_option('--output-dir', type=str, default='output/',
                      help='Output directory from split_data.py: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    transformers_dir = options.transformers_dir
    start_seed = options.seed
    folds = options.folds
    max_seq_length = options.max_seq_length
    save_steps = options.save_steps
    n_seeds = options.n_seeds
    first_seed = options.first_seed
    last_seed = options.last_seed
    device = options.device
    output_dir = options.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(start_seed)
    seeds = np.random.randint(low=0, high=2**32 - 1, size=n_seeds)

    for do_text_b in [False, True]:
        for seed in seeds[first_seed:last_seed+1]:
            for weights in [False, True]:
                for model in ['bert-base-uncased']:
                    for lr in [1e-5, 2e-5, 4e-5]:
                        run_folds(basedir=basedir,
                                  transformers_dir=transformers_dir,
                                  seed=seed,
                                  model_name_or_path=model,
                                  folds=folds,
                                  start_fold=0,
                                  lr=lr,
                                  max_seq_length=max_seq_length,
                                  do_text_b=do_text_b,
                                  weights=weights,
                                  save_steps=save_steps,
                                  exp_dir=output_dir,
                                  device=device)


if __name__ == '__main__':
    main()
