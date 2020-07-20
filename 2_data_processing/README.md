# Data processing

[Intro blurb] 

## Extracting embedded opinion clauses from article texts

`1_extract_quotes.py` implements our dependency parse-based algorithm for extracting embedded [Opinion]{.smallcaps} spans (e.g. *Scientists believe that [**climate change requires immediate action**]*), [Sources]{.smallcaps} (e.g. *Scientists*) and [Predicates]{.smallcaps} (e.g. *believe*) from a given article. 

Sample usage:

```
python 1_extract_quotes.py \
	--debug \ 									# whether to test run on smaller sample
	--input_df_filename ../1_data_scraping/output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl \ # where to read in dataset df
	--output_dir url_quotes \ 							# where to write jsons with extracted annotations
	--fulltext_dir url_texts 							# where to read in article full texts
```

Running the above writes a `json` for every article with .

## Deduplicating articles

`1_dedup_titles.py` deduplicates articles from the same media outlet based on their publish dates and the edit distance of their associated titles, after regularization. You can use the deduplicated output we provide (`output/dedup_combined_df_2000_1_1_to_2020_4_12.pkl`) directly, without running the deduplication script (which can be slow). 

If you do wish to run the script, you can use the article meta-data we provide (`output/temp_combined_df_2000_1_1_to_2020_4_12.pkl`) or your own output from `0_get_urls.py` as input. 

Sample usage:
```
python 1_dedup_tites.py \
	--input_df_filename output/temp_combined_df_2000_1_1_to_2020_4_12.pkl \	# where to read in pre-deduplicated data
	--output_df_filename dedup_combined_df_2000_1_1_to_2020_4_12.pkl	# where to save deduplicated data
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