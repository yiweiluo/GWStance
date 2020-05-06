# Data scraping module

`0_get_urls.py` makes use of two separate APIs to fetch URLs of news articles related to climate change:
* the MediaCloud Python client (https://github.com/mitmedialab/MediaCloud-API-Client), 
* SerpApi (https://serpapi.com/search-api), which scrapes Google search results from querying climate change-related keywords on various websites

To run `0_get_urls.py`, you need to get API keys for both. You can register for a MediaCloud API key for free; SerpAPI is only free within a limited trial period. Once you have gotten API keys, save them to files named 'MC_API_KEY.txt' and 'SERP_API_KEY.txt', respectively, in your local copy of this directory.