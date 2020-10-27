import os
from optparse import OptionParser

import numpy as np
import pandas as pd
import pystan


# try a categorical logit model  (i.e. multinomial with logged param vector)
model = """
data { 
  int<lower=1> n_questions;
  int<lower=1> n_workers;
  int<lower=1> n_ratings;
  int<lower=1, upper=n_workers> worker_for_rating[n_ratings];
  int<lower=1, upper=n_questions> question_for_rating[n_ratings];
  real q_priors[3];
  //real worker_attribs[n_workers];
  int ratings[n_ratings];
}
parameters {
  vector[n_questions] q_position_low;
  vector[n_questions] q_position_mid;  
  vector[n_questions] q_position_high;
  real q_mean_low;
  real q_mean_mid;
  real q_mean_high;
  real q_overall_mean;
  real<lower=0> q_std;
  vector[n_workers] worker_offsets_1;
  vector[n_workers] worker_offsets_2;
  vector[n_workers] worker_offsets_3;
  real<lower=0, upper=1> prop_rand[n_workers];
  real<lower=0> offset_std;
  //real attrib_effects_low;
  //real attrib_effects_mid;
  //real attrib_effects_high;  
  //real<lower=0> effects_std;
  //ordered[2] agreement_offsets;
}
model {
  // Priors
  q_std ~ normal(0, 1);
  //q_position_low ~ normal(-1.67, q_std);
  //q_position_mid ~ normal(-0.84, q_std);  
  //q_position_high ~ normal(-0.96, q_std);
  q_position_low ~ normal(q_priors[1], q_std);
  q_position_mid ~ normal(q_priors[2], q_std);  
  q_position_high ~ normal(q_priors[3], q_std);
  
  // Multilevel model
  offset_std ~ normal(0, 1);  
  for (w in 1:n_workers) {
    //worker_offsets[w] ~ normal(worker_attribs[w] * attrib_effects, offset_std);
    //worker_offsets_1[w] ~ normal(worker_attribs[w] * attrib_effects_low, offset_std);
    //worker_offsets_3[w] ~ normal(worker_attribs[w] * attrib_effects_high, offset_std);
    worker_offsets_1[w] ~ normal(0, offset_std);
    worker_offsets_2[w] ~ normal(0, offset_std);
    worker_offsets_3[w] ~ normal(0, offset_std);

  }  
  for (r in 1:n_ratings) {
    vector[3] logits;
    logits[1] = (1-prop_rand[worker_for_rating[r]]) * q_position_low[question_for_rating[r]] + prop_rand[worker_for_rating[r]] * worker_offsets_1[worker_for_rating[r]];
    logits[2] = (1-prop_rand[worker_for_rating[r]]) * q_position_mid[question_for_rating[r]] + prop_rand[worker_for_rating[r]] * worker_offsets_2[worker_for_rating[r]];
    logits[3] = (1-prop_rand[worker_for_rating[r]]) * q_position_high[question_for_rating[r]] + prop_rand[worker_for_rating[r]] * worker_offsets_3[worker_for_rating[r]];
    ratings[r] ~ categorical_logit(logits);  
  }
}
"""


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--iter', type=int, default=4000,
                      help='Number of iterations: default=%default')
    parser.add_option('--chains', type=int, default=5,
                      help='Number of chains: default=%default')

    (options, args) = parser.parse_args()

    response_df = pd.read_csv('data/all_responses_updated.tsv', header=0, index_col=0, sep='\t')
    response_df.head()

    sents_df = pd.read_csv('data/sents.tsv', header=0, index_col=0, sep='\t')
    sents_df.head()

    ratings = np.array(response_df['ratings'].values)
    worker_indices = response_df['worker_indices'].values
    question_indices = response_df['question_indices'].values

    counts = np.zeros(3)
    for r in ratings:
        counts[r] += 1
    props = counts / counts.sum()
    priors = np.log(props)
    print(props)
    print(priors)

    data = {'n_questions': 2050,
            'n_workers': 400,
            'n_ratings': len(ratings),
            'worker_for_rating': [worker_indices[i] + 1 for i in range(len(ratings))],
            'question_for_rating': [question_indices[i] + 1 for i in range(len(ratings))],
            'ratings': [int(ratings[i]) + 1 for i in range(len(ratings))],
            'q_priors': list(priors)
            }

    sm = pystan.StanModel(model_code=model)
    fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

    low = fit.extract('q_position_low')['q_position_low']
    mid = fit.extract('q_position_mid')['q_position_mid']
    high = fit.extract('q_position_high')['q_position_high']

    low_mean = np.exp(low.mean(0))
    mid_mean = np.exp(mid.mean(0))
    high_mean = np.exp(high.mean(0))

    print(low_mean, mid_mean, high_mean)

    unnorm_probs = np.vstack([low_mean, mid_mean, high_mean]).T
    n_questions, n_cols = unnorm_probs.shape
    row_sums = unnorm_probs.sum(1)
    probs = unnorm_probs / row_sums.reshape((n_questions, 1))
    print(probs.shape)
    predictions = np.argmax(probs, 1)

    labels = {0: 'disagree', 1: 'neutral', 2: 'agree'}

    sents_df['disagree'] = probs[:, 0]
    sents_df['neutral'] = probs[:, 1]
    sents_df['agree'] = probs[:, 2]
    sents_df['label'] = [labels[i] for i in predictions]

    sents_df.to_csv(os.path.join('data', 'sent_scores_df.tsv'), sep='\t')


if __name__ == '__main__':
    main()
