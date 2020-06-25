# Data details

Out of copyright concerns, we do not make the full text of the corresponding URLs, nor the code for scraping the text of the URLs available. However, we release the full set of URLs and corresponding meta information (article headline, publish date, etc.) in `dedup_df.tsv`. 

* `0_get_urls.py` makes use of two separate APIs to fetch URLs of news articles related to climate change:
	* the MediaCloud Python client (https://github.com/mitmedialab/MediaCloud-API-Client), 
	* SerpApi (https://serpapi.com/search-api), which scrapes Google search results from querying climate change-related keywords on various websites

	To run `0_get_urls.py`, you need to get API keys for both. You can register for a MediaCloud API key for free; SerpAPI is only free within a limited trial period. Once you have gotten API keys, either copy them to the respective fields in `config.json` (in the root directory), or save them to files named 'MC_API_KEY.txt' and 'SERP_API_KEY.txt', respectively, in your local copy of this sub-directory.

	You can use the default news domains that we used to retrieve news stories (`google_domains.txt` contains domains we fetched from using SerpApi, `mediacloud_ids.txt` contains domains we fetched from with the MediaCloud API), or you can modify these text files to add/remove domains as you like. Note, however, that MediaCloud requires an ID corresponding to a news domain to fetch from it (see their instructions [here](https://github.com/berkmancenter/mediacloud/blob/master/doc/api_2_0_spec/api_2_0_spec.md#grab-all-stories-in-the-new-york-times-during-october-2012)).

Sample usage:


* `1_dedup_titles.py` regularizes the headlines of fetched URLs for deduplication purposes.