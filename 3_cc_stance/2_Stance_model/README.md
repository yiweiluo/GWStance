This README file provides information about scripts included with this repo for doing label inference from the noisy annotator labels, running the demographic models to look for differences between groups, applying the pre-trained climate stance model to new data, and running a new BERT model search.

### 1. Label inference

Label inference is done using Stan. To set up an environment for this, first create a new environment using `conda create -n stan python=3`, activate it using `conda activate stan`, and then install the required packages using `conda install numpy scipy pandas pystan -c conda-forge`.

One script is provided here, along with data, to replicate the label inference from the raw data. To run, use: `python infer_labels.py`. This should call Stan and run the sampler. To test this without doing full inference, it can be run using fewer steps, e.g., by adding `--iter 200`, though defaults should be used for replication.

### 2. Demographic Models

Scripts are similarly provided to replicate the demographic analyses reported in the paper. These have the same requirements as above (numpy, scipy, pandas, pystan), and have been split into single variable models, and multivariate models. For the former, run `python run_demographic_single_var_model.py`. For the latter, run `python run_demograhpic_multi_var_model.py`. Both of these will take some time to run each model.


### 3. Applying trained stance model

To apply the final model stance model used in the paper, it is necessary to first clone and install the [transformers](https://github.com/huggingface/transformers) library, and then copy some additional scripts into its directory. To do so, use the following steps:

1. Create a new conda environment (i.e., `conda create -n climate python=3`
2. Activate it using `conda activate climate`
3. Install pytorch by following the directions on its [website](https://pytorch.org/get-started/locally/), e.g, `conda install pytorch torchvision cudatoolkit=9.2 -c pytorch`
4. Install additional packages using `conda install scipy pandas matplotlib scikit-learn tqdm tensorboard boto3`
5. Install `transformers` from the source by following [these instructions](https://huggingface.co/transformers/installation.html#installing-from-source)
6. Test the installation to make sure you can load and run a basic BERT model using the instructions on the huggingface README.
7. Copy the files from the `for_transformers` directory in this repo using the following command (using your path to the `transformers` directory): `cp -r for_transformers/* </path/to/transformers/>`
8. Download the pretrained climate stance model from [here](https://drive.google.com/file/d/12rVg_bpuDfZbdWRtEN2Jf6SNyMEnax76/view?usp=sharing) (~400 Mb) and extract it the same directory as this `README.md` file using `tar -xvzf final_model.tar.gz`
9. Format your data in .tsv format, such that each line is a document, structured as `[text][\t][label][\t][1.0]`, where `[text]` is a string without tabs or line breaks, `[label]` should be a dummy label from \{agree, disagree, neutral\} and 1.0 is a weight. Save it in any directory as `test.tsv`
10. Do the prediction using `python predict.py final_model/config.json final_model/no-dev/ --data-dir </path/to/data/> --transformers-dir </path/to/transformers/>` where `</path/to/data/>` is the path to the directory containing your `test.tsv` file, and `</path/to/transformers/>` is the path to your cloned transformers dir.
11. The output will be written to `final_model/no-dev/predictions_test.tsv`

### 4. Running BERT model search

To replicate the model search used in the paper, it is necessary to first set up the environment as above. To do so, use the following steps:

First, follow the steps from `3. Applying trained stance model` above, up to (and including) step 7. Then,

1. Create the necessary data splits using `python split_data.py`, which should create a directory called `splits`
2. Test that you can run one model using `python run_folds.py --folds 1 --transformers-dir /path/to/transformers/` (again using your path to the transformers dir)
3. If that works, run the full search using `python run_search.py --transformers-dir </path/to/transformers/>`
4. The scripts `summarize_runs.py`, `rerun_best.py`, and `predict.py` can also be used to identify the best configuration (on validation data), and then train a new model using that configuration on all non-test data, and evaluate it on the test set.
