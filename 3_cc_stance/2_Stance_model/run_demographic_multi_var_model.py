import os
from optparse import OptionParser

import pystan
import numpy as np
import pandas as pd
from scipy.special import expit


# Multilevel logistic model with a multiple covariates
multi_var_model = """
data { 
  int<lower=1> n_questions;
  int<lower=1> n_workers;
  int<lower=1> n_ratings;
  int<lower=1> n_covariates;
  int<lower=1, upper=n_workers> worker_for_rating[n_ratings];
  int<lower=1, upper=n_questions> question_for_rating[n_ratings];
  vector[n_covariates] covariates[n_workers];
  int ratings[n_ratings];
}
parameters {
  real<lower=0> question_std;
  vector[n_questions] q_position;

  real<lower=0> worker_offset_std;
  vector[n_workers] worker_offsets;
  
  vector[n_covariates] fixed_effects;

  ordered[2] agreement_offsets;
} 
model {
  // Priors
  question_std ~ normal(0, 1);
  q_position ~ normal(0, question_std);
  
  worker_offset_std ~ normal(0, 1);
  
  for (w in 1:n_workers) {
    worker_offsets[w] ~ normal(dot_product(covariates[w], fixed_effects), worker_offset_std);
  }  
  for (r in 1:n_ratings) {
    ratings[r] ~ ordered_logistic(q_position[question_for_rating[r]] + worker_offsets[worker_for_rating[r]], agreement_offsets);  
  }
}
"""


def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--chains', type=int, default=5,
                      help='Number of chains: default=%default')
    parser.add_option('--iter', type=int, default=3000,
                      help='Number of iterations: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    basedir = args[0]

    chains = options.chains
    iterations = options.iter

    data_dir = 'data'

    worker_df = pd.read_csv(os.path.join(data_dir, 'worker_attributes.tsv'), header=0, index_col=0, sep='\t')
    response_df = pd.read_csv(os.path.join(data_dir, 'all_responses_updated.tsv'), header=0, index_col=0, sep='\t')

    covariate_groups = {'all_demographic': ['over34', 'female', 'college_plus', 'republican', 'democrat'],
                        'party': ['republican', 'democrat'],
                        'republican_and_female': ['republican', 'female'],
                        'party_and_gender': ['republican', 'democrat', 'female']
                        }

    for name, group in covariate_groups.items():

        print("Running", name)

        data = {'n_questions': len(set(response_df['question_indices'])),
                'n_workers': len(worker_df),
                'n_ratings': len(response_df),
                'n_covariates': len(group),
                'worker_for_rating': np.array(response_df['worker_indices'].values, dtype=int) + 1,
                'question_for_rating': np.array(response_df['question_indices'].values, dtype=int) + 1,
                'covariates': np.array(worker_df[group].values, dtype=int),
                'ratings': np.array(response_df['ratings'].values, dtype=int) + 1
                }

        sm = pystan.StanModel(model_code=multi_var_model)
        fit = sm.sampling(data=data, iter=iterations, chains=chains)

        fixed_effect_samples = fit.extract('fixed_effects')['fixed_effects']
        q_position_samples = fit.extract('q_position')['q_position']
        worker_offsets_samples = fit.extract('worker_offsets')['worker_offsets']
        agreement_offsets_samples = fit.extract('agreement_offsets')['agreement_offsets']

        outdir = os.path.join(basedir, name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        np.savez(os.path.join(outdir, 'group_effect_samples.npz'),
                 fixed_effects=fixed_effect_samples,
                 covariates=group,
                 q_position_mean=q_position_samples.mean(1),
                 worker_offset_mean=worker_offsets_samples.mean(1),
                 agreement_offsets=agreement_offsets_samples)

        # Average across questions
        q = q_position_samples.mean(1)
        # Average across workers
        w = worker_offsets_samples.mean(1)
        t = agreement_offsets_samples
        f = fixed_effect_samples
        # get the agreement ratio for the covariate for each sample (to get mean and CI)

        print("Covariate effects (with 95% CIs):")
        for c_i, covariate in enumerate(group):
            agree_ratio = expit(q + w + f[:, c_i] + t[:, 1]) / expit(q + w + t[:, 1])
            print(covariate, 'agree ratio (with 95% CI): {:.3f} ({:.3f}, {:.3f})'.format(np.mean(agree_ratio), np.percentile(agree_ratio, q=2.5), np.percentile(agree_ratio, q=97.5)))


if __name__ == '__main__':
    main()
