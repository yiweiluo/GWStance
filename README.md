# **De**tecting **S**tance in **M**edia **o**n **G**lobal warming

This repository contains code and data for the paper:
> Luo, Y., Card, D. and Jurafsky, D. (2020). Detecting Stance in Media on Global Warming. In *Findings of the Association for Computational Linguistics: EMNLP 2020*.
```
@inproceedings{luo-etal-2020-detecting,
    title = "Detecting Stance in Media On Global Warming",
    author = "Luo, Yiwei  and
      Card, Dallas  and
      Jurafsky, Dan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.296",
    doi = "10.18653/v1/2020.findings-emnlp.296",
    pages = "3296--3315",
    abstract = "Citing opinions is a powerful yet understudied strategy in argumentation. For example, an environmental activist might say, {``}Leading scientists agree that global warming is a serious concern,{''} framing a clause which affirms their own stance ({``}that global warming is serious{''}) as an opinion endorsed (''[scientists] agree{''}) by a reputable source ({``}leading{''}). In contrast, a global warming denier might frame the same clause as the opinion of an untrustworthy source with a predicate connoting doubt: {``}Mistaken scientists claim [...].'' Our work studies opinion-framing in the global warming (GW) debate, an increasingly partisan issue that has received little attention in NLP. We introduce DeSMOG, a dataset of stance-labeled GW sentences, and train a BERT classifier to study novel aspects of argumentation in how different sides of a debate represent their own and each other{'}s opinions. From 56K news articles, we find that similar linguistic devices for self-affirming and opponent-doubting discourse are used across GW-accepting and skeptic media, though GW-skeptical media shows more opponent-doubt. We also find that authors often characterize sources as hypocritical, by ascribing opinions expressing the author{'}s own view to source entities known to publicly endorse the opposing view. We release our stance dataset, model, and lexicons of framing devices for future work on opinion-framing and the automatic detection of GW stance.",
}
```

## Getting started
1. Create and activate a Python 3.6 environment.
2. Run `pip install -r requirements.txt`.
3. Re-install neuralcoref with the `--no-binary` option: 
```
pip uninstall neuralcoref
pip install neuralcoref --no-binary neuralcoref
```
4. Download SpaCy's English model: `python -m spacy download en`
5. Update the `config.json` file with your local OS variables.

## Repository structure

* Our **dataset GWSD** itself can be accessed via `GWSD.tsv` in the main directory. The dataset contains tab-separated fields for each of the following:
	1. `sentence`: the sentence 
	2. `worker_0`, ..., `worker_7`: ratings from each of the 8 workers for the stance of the sentence
	3. `disagree`: the probability that the sentence expresses disagreement with the target (that climate change/global warming is a serious concern), as estimated by our Bayesian model
	4. `agree`: ditto for the "agrees" label
	5. `neutral`: ditto for the "neutral" label
	6. `guid`: a unique ID for each sentence
	7. `in_held_out_test`: whether the sentence was used in our held-out-test set for model and baseline evaluation

Note: The first 5 rows are the 5 screen sentences we use to make sure that annotators correctly understand the task, and thus do not have estimated probability distributions.
* Our **lexicons of framing devices** are located in `4_analyses/lexicons`.
* The sequence of code to replicate our results can be found in the individual READMEs of the numbered sub-directories.
