# Data details

Out of copyright concerns, we do not make the full text of the corresponding URLs, nor the code for scraping the text of the URLs available. However, we release the full set of URLs and corresponding meta information (article headline, publish date, etc.) in `dedup_df.tsv`. 

* `0_get_urls.py` makes use of two separate APIs to fetch URLs of news articles related to climate change:
	* the MediaCloud Python client (https://github.com/mitmedialab/MediaCloud-API-Client), 
	* SerpApi (https://serpapi.com/search-api), which scrapes Google search results from querying climate change-related keywords on various websites

	To run `0_get_urls.py`, you need to get API keys for both. You can register for a MediaCloud API key for free; SerpAPI is only free within a limited trial period. Once you have gotten API keys, either copy them to the respective fields in `config.json` (in the root directory), or save them to files named 'MC_API_KEY.txt' and 'SERP_API_KEY.txt', respectively, in your local copy of this sub-directory.


* `dedup_titles.py` regularizes the headlines of fetched URLs for deduplication purposes.