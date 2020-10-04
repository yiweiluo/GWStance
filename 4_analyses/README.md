# Analysis

Scripts and helper files for doing opinion-framing analyses as described in our *Findings* paper. The `lexicons` directory contains individual `.txt` files for affirming and doubting framing devices.

## Gathering model predictions

`1_gather_preds.py` adds the stances for Opinion spans as labeled by our weighted BERT model to a dataframe with different columns for framing context, article meta-information, etc. for ease of analyzing opinion-framing patterns.

## Opinion-framing studies

`2_opinion_framing.py` runs the analysis scripts.