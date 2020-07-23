# Data processing

[Intro blurb] 

## Extracting embedded opinion clauses from article texts

`1_extract_quotes.py` implements our dependency parse-based algorithm for extracting embedded [Opinion]{.smallcaps} spans (e.g. *Scientists believe that [**climate change requires immediate action**]*), [Sources]{.smallcaps} (e.g. *Scientists*) and [Predicates]{.smallcaps} (e.g. *believe*) from a given article. 

To run this script, you will need:
1. A dataframe containing the set of articles you want to analyze, for which you have two options:
	* Option A: The original dataset we use in the paper, provided as `curr_dedup_df.tsv` in `1_data_scraping`;
	* Option B: A different set that you collect from running the scripts provided in `1_data_scraping`.
	The path to this dataframe will be specified via the `--input_df_filename` command line argument.
1. The full text of each article saved as a `.txt` file, named according to the convention `'url_no_{}.txt'.format(guid)` (e.g. `url_no_7.txt`), where `guid` is a global unique identifier indexed to each article and stored as its own column in the article dataframe. The path to the directory containing the text files will be specified via the `--fulltext_dir` command line argument.

Sample usage:

```
python 1_extract_quotes.py \
	--debug \ 					# whether to test run on smaller sample 
	--input_df_filename ../1_data_scraping/output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl \ 	# where to read in article dataset df
	--output_dir url_quotes \ 			# where to write jsons with extracted annotations
	--fulltext_dir url_texts 			# where to read in article full texts
```

Running the above writes a `.json` (using the naming convention `'url_no_{}.json'.format(guid)`) for every article with the following structure:

```yaml
{
   "quote_tags": {
      "0": {										# index of sentence within article, as a `str`
         "idx2text": {"0": "Almost", "1": "no", "2": "rational", "3": "people", ... }, 	# dict mapping each token's index within the document to the token's text
         "idx2lemma": {"0": "almost", "1": "no", "2": "rational", "3": "person", ...},  # dict mapping each token's index within the document to the token's lemmatized text
         "quotes": [     								# list of dicts containing annotations for all (Source, Predicate, Opinion) tuples (plus additional modifiers) that occur in the sentence
                     {
                        "neg_s": [0, 1],						# indices of negation tokens modifying the Source (e.g. "**Almost no** rational people would point out that climate change is a hoax.")
                        "main_neg_s": [1],						# index of the head negation token modifying the Source (e.g. "Almost **no** rational people would point out that climate change is a hoax.")    
                        "s": [2, 3],							# indices of Source tokens (e.g. "Almost no **rational people** would point out that climate change is a hoax.")
                        "main_s": [3],							# index of the head Source token (e.g. "Almost no rational **people** would point out that climate change is a hoax.")		
                        "neg_v": [],							# indices of the Predicate negation tokens
                        "main_neg_v": [],						# index of the head Predicate negation token
                        "v": [4, 5, 6],							# indices of the Predicate tokens (e.g. "Almost no rational people **would point out** that climate change is a hoax.")
                        "main_v": [5],							# index of the head Predicate token (e.g. "Almost no rational people would **point** out that climate change is a hoax.")
                        "v_prt": [6],							# indices of tokens that are particles attached to the Predicate (e.g. "Almost no rational people would point **out** that climate change is a hoax.")
                        "q": [7, 8, 9, 10, 11, 12, 13, 14]				# indices of tokens that are part of the embedded Opinion (e.g. "Almost no rational people would point out **that climate change is a hoax.**")
                     },
                     { 
                        ...
                     }
         ]        
      },
      "1": {
              ...
      }
   },
   "coref_tags": {									# dict mapping the index of each token in the document to its co-refering string, if present
        "0": null,
        "1": null,
        "2": null,
        ...
   }
}
```

## Filtering extracted Opinions

`2_filter_quotes.py` applies the following criteria to filter extracted opinions:
1. The extracted Predicate must be a Householder verb,
2. The extracted Opinion cannot be an indirect question,
3. The extracted Opinion must contain one of 71 climate change/global warming-related keywords. 

Sample usage:
```
python 1_dedup_tites.py \
	--path_to_df ../1_data_scraping/curr_dedup_df.tsv \	# where to read in dataframe of articles
	--output_dir ./output \					# where to write filtered data
	--quotes_dir ./new_extracted_quotes			# directory containing extracted Opinions
```

## Filtering articles

You can use the pre-filtered data we provide (`output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl`) or run `2_filter_dataset.py` on your own input dataset.

`2_filter_dataset.py` finds articles from the input dataset that have an associated publication date and that have non-empty article text. To use this script, you need to first have the full article texts of the articles in your dataset scraped and saved as individual `.txt` files with the filename adhering to the format `url_no_N`, where `N` is a global unique ID associated with each article. (Out of copyright concerns, we do not make the full text of the dataset we provide available.) 

Sample usage:
```
python 2_filter_dataset.py \
	--input_df_filename output/dedup_combined_df_2000_1_1_to_2020_4_12.pkl \	# where to read in dataset
	--output_df_filename filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl \	# where to save filtered data
	--url_text_dir url_texts 							# where to find the scraped article full texts
```

## Exploring article stats

`3_explore_dataset.py` creates a report of the basic meta-information obtained for all articles in an dataset.

Sample usage:
```
python 3_explore_dataset.py \
	--input_data_filename  output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl \	# where to read in data
```