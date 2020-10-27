# **DeSMOG**: **De**tecting **S**tance in **M**edia **O**n **G**lobal warming

This repository contains code and data for the paper:
> Luo, Y., Card, D. and Jurafsky, D. (2020). DeSMOG: Detecting Stance in Media On Global Warming. In *Findings of EMNLP*.
```
@article { luo-et-al-2020,
	   author = "Yiwei Luo, Dallas Card and Dan Jurafsky",
	   title = "DeSMOG: Detecting Stance in Media On Global Warming",
	   journal = Findings of EMNLP,
	   year = 2020,
	   pages = ,
}
```

## Getting started
1. `pip install -r requirements.txt` (and make sure you are using Python 3.6)
2. Re-install neuralcoref with the `--no-binary` option: 
```
pip uninstall neuralcoref
pip install neuralcoref --no-binary neuralcoref
```
3. Download SpaCy's (small) English model: `python -m spacy download en`
4. Update the `config.json` file.

## Repository structure

* Our dataset DeSMOG itself can be accessed via `desmog.tsv` in the main directory. The dataset contains tab-separated fields for each of the following:
	1. `sentence`: the sentence 
	2. `worker_0`, ..., `worker_7`: ratings from each of the 8 workers for the stance of the sentence
	3. `disagree`: the probability that the sentence expresses disagreement with the target (that climate change/global warming is a serious concern), as estimated by our Bayesian model
	4. `agree`: ditto for the "agrees" label
	5. `neutral`: ditto for the "neutral" label
	6. `guid`: a unique ID for each sentence
	7. `in_held_out_test`: whether the sentence was used in our held-out-test set for model and baseline evaluation

Note: The first 5 rows are the 5 screen sentences we use to make sure that annotators correctly understand the task, and thus do not have estimated probability distributions.
* Our lexicons of framing devices are located in `4_analyses/lexicons`.
* The sequence of code to replicate our results can be found in the individual READMEs of the numbered sub-directories.

## Notes:
* This repo 