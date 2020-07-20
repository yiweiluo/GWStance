# Data details

This directory contains `curr_dedup_df.tsv`, the dataframe containing URLs and meta-data on the full set of unique articles (title, publish date, outlet source) that we use in our analysis. Out of copyright concerns, we do not make the full text of the corresponding URLs, nor the code for scraping the text of the URLs available. 

We do provide scripts and helper files for obtaining and deduplicating additional article URLs, if you want to get your own data from different time ranges, using different keyword search terms, etc. 

## Getting URLs of climate change articles

`0_get_urls.py` makes use of two separate APIs to fetch URLs of news articles related to climate change:
* the MediaCloud Python client (https://github.com/mitmedialab/MediaCloud-API-Client), 
* SerpApi (https://serpapi.com/search-api), which scrapes Google search results from querying climate change-related keywords on various websites

To run `0_get_urls.py`, you need to get API keys for both. You can register for a MediaCloud API key for free; SerpAPI is only free within a limited trial period. Once you have gotten API keys, either copy them to the respective fields in `config.json` (in the root directory), or save them to files named 'MC_API_KEY.txt' and 'SERP_API_KEY.txt', respectively, in your local copy of this sub-directory.

You can use the default news domains that we used to retrieve news stories (`google_domains.txt` contains domains we fetched from using SerpApi, `mediacloud_ids.txt` contains domains we fetched from with the MediaCloud API), or you can modify these text files to add/remove domains as you like. Note, however, that MediaCloud requires an ID corresponding to a news domain to fetch from it (see their instructions [here](https://github.com/berkmancenter/mediacloud/blob/master/doc/api_2_0_spec/api_2_0_spec.md#grab-all-stories-in-the-new-york-times-during-october-2012)).

Sample usage:

```
python 0_get_urls.py \
	--do_serp \ 			# whether to run SerpAPI URL retrieval
	--do_mediacloud \		# whether to run MediaCloud URL retrieval
	--mediacloud_start_year 2019 \ 	# start year threshold for MediaCloud (default 2000)
	--mediacloud_start_month 12 \	# start month threshold for MediaCloud (default 1)
	--mediacloud_start_day 1 \	# start day threshold for MediaCloud (default 1)
	--mediacloud_end_year 2020 \	# end year threshold for MediaCloud (default 2020)
	--mediacloud_end_month 4 \	# end month threshold for MediaCloud (default 12)
	--mediacloud_end_day 12		# end day threshold for MediaCloud (default 31)
```

Running the above creates intermediary files storing the results of fetching stories with SerpAPI and MediaCloud, as well as a dataframe containing the combined results with article URLs and meta-information (`output/temp_combined_df_2000_1_1_to_2020_4_12.pkl`).

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