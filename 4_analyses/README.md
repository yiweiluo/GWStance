# Analysis

Scripts and helper files for doing opinion-framing analyses as described in our *Findings* paper. The `lexicons` directory contains individual `.txt` files for affirming and doubting framing devices.

Sample usage:
```
python opinion_framing.py --path_to_input curr_output/quote_analysis_df.pkl   # this dataframe contains Opinion spans as labeled by our weighted BERT model along with columns for framing context, article meta-information, etc. 
			    --do_subsample 				      # whether to do analyses additionally on data that excludes articles from the top 5 Right- and Left-leaning outlets
			    --do_nonwire				      # whether to do analyses additionally on data that excludes articles from news wires
```

