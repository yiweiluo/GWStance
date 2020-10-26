# Analysis

Scripts and helper files for doing opinion-framing analyses as described in our *Findings* paper. The `lexicons` directory contains individual `.txt` files for affirming and doubting framing devices.

Running `python 0_process_predictions.py` will gather the batched model prediction output with stance labels for each extracted quote, then annotate these quotes for information like the framing context (Source, Predicate, modals, negation) and the outlet context (media slant, media domain, publish date, etc.) to enable the analysis of patterns in opinion attribution. This repo contains the predicted output of our model in `3_cc_stance/2_Classifier/BERT_preds`, but you can also read in predictions for different data by changing the input path. 

`1_opinion_framing.py` uses the output from `0_process_predictions.py` to generate plots and summaries of opinion-framing patterns.

Sample usage:
```
python 1_opinion_framing.py --path_to_input curr_output/quote_analysis_df.pkl  # this dataframe contains Opinion spans as labeled by our weighted BERT model along with columns for framing context, article meta-information, etc. 
			  --do_subsample 				       # whether to do analyses additionally on data that excludes articles from the top 5 Right- and Left-leaning outlets
			  --do_nonwire				      	       # whether to do analyses additionally on data that excludes articles from news wires
```

