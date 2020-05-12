import os
import pandas as pd
import numpy as np
import re

quotes_df = pd.read_csv('keyword_filtered_comp_clauses.tsv',sep='\t',header=0,index_col=0)
