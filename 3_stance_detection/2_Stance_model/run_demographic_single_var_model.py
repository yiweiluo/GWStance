import os
from optparse import OptionParser

import pystan
import numpy as np
import pandas as pd
from scipy.special import expit


# Multilevel logistic model with a single covariate
single_var_model = """
data {
  int<lower=1> n_questions;
  int<lower=1> n_workers;
  int<lower=1> n_ratings;
  int<lower=1, upper=n_workers> worker_for_rating[n_ratings];
  int<lower=1, upper=n_questions> question_for_rating[n_ratings];
  int<lower=0, upper=1> group[n_workers];
  int ratings[n_ratings];
}
parameters {
  real<lower=0> question_std;
  vector[n_questions] q_position;

  real<lower=0> worker_offset_std;
  vector[n_workers] worker_offsets;
  real group_effect;

  ordered[2] agreement_offsets;
}
model {
  // Priors
  question_std ~ normal(0, 1);
  q_position ~ normal(0, question_std);

  worker_offset_std ~ normal(0, 1);

  //group_effect ~ normal(0, 2);

  for (w in 1:n_workers) {
    worker_offsets[w] ~ normal(group_effect * group[w], worker_offset_std);
  }
  for (r in 1:n_ratings) {
    ratings[r] ~ ordered_logistic(q_position[question_for_rating[r]] + worker_offsets[worker_for_rating[r]], agreement_offsets);
  }
}
"""


def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_argument('--base_dir', type=str, default="output",
                      help='Base directorys: default=%default')
    parser.add_option('--chains', type=int, default=5,
                      help='Number of chains: default=%default')
    parser.add_option('--iter', type=int, default=3000,
                      help='Number of iterations: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()
    print('options:',options)
    print('args:',args)
    basedir = args[0]

    chains = options.chains
    iterations = options.iter

    data_dir = 'data'

    worker_df = pd.read_csv(os.path.join(data_dir, 'worker_attributes.tsv'), header=0, index_col=0, sep='\t')
    response_df = pd.read_csv(os.path.join(data_dir, 'all_responses_updated.tsv'), header=0, index_col=0, sep='\t')

    covariates = ['over34', 'male', 'female', 'college_plus', 'republican', 'democrat']

    for covariate in covariates:

        print("Running", covariate)

        data = {'n_questions': len(set(response_df['question_indices'])),
                'n_workers': len(worker_df),
                'n_ratings': len(response_df),
                'worker_for_rating': np.array(response_df['worker_indices'].values, dtype=int) + 1,
                'question_for_rating': np.array(response_df['question_indices'].values, dtype=int) + 1,
                'group': np.array(worker_df[covariate].values, dtype=int),
                'ratings': np.array(response_df['ratings'].values, dtype=int) + 1
                }

        sm = pystan.StanModel(model_code=single_var_model)
        fit = sm.sampling(data=data, iter=iterations, chains=chains)

        group_effect_samples = fit.extract('group_effect')['group_effect']

        q_position_samples = fit.extract('q_position')['q_position']
        worker_offsets_samples = fit.extract('worker_offsets')['worker_offsets']
        agreement_offsets_samples = fit.extract('agreement_offsets')['agreement_offsets']

        outdir = os.path.join(basedir, covariate)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        np.savez(os.path.join(outdir, 'group_effect_samples.npz'),
                 group_effect=group_effect_samples,
                 q_position_mean=q_position_samples,
                 worker_offset_mean=worker_offsets_samples,
                 agreement_offsets=agreement_offsets_samples)

        # Average across questions
        q = q_position_samples.mean(1)
        # Average across workers
        w = worker_offsets_samples.mean(1)
        e = group_effect_samples
        t = agreement_offsets_samples
        # get the agreement ratio for the covariate for each sample (to get mean and CI)
        agree_ratio = expit(q + w + e + t[:, 1]) / expit(q + w + t[:, 1])
        print(covariate, 'agree ratio (with 95% CI): {:.3f} ({:.3f}, {:.3f})'.format(np.mean(agree_ratio), np.percentile(agree_ratio, q=2.5), np.percentile(agree_ratio, q=97.5)))


if __name__ == '__main__':
    main()
