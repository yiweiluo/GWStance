# Climate change stance classifier

TODOs

- [ ] Training and test data
   - [ ] SemEval 2016 task 6 train, trial--both very small (211 for, 15 against)
   - [ ] SemEval 2016 test (subtask A)--169 tweets (123 for, 11 against, 35 none)
   - [ ] Mohammad et al. paper--data requested
   - [ ] Dallas' annotated news data
   - [ ] Yiwei's news data--needs annotation
- [ ] MTurk 
   - [x] pilot, iterate with smaller subsample to check inter-annotator agreement
   - [ ] Lock down non-pilot specifics:
	- [ ] Config settings: 97% minPercentPreviousHITsApproved, 1000 minNumPreviousHITsApproved, USonly = yes
	- [ ] Collect 2,000 total annotations in 5 rounds: N=300, 400, 400, 450, 450
	- [ ] 6 annotators per annotation item
	- [ ] Each HIT will have N/10 true items to annotate + 5 screen items (screen items differ b/w rounds, so need 25 total) (so Round 1 will have 35; Round 2, 45; Round 3, 50)
	- [ ] N items/(N/10 items/HIT) = 10 HITs per round; multiply by 6 annotators = 60 annotators paid
	- [ ] Payment (based on $12/hr MW in CA): $4, $5.14, $5.7 for Rounds 1, 2, 3, 4, 5; total cost = $4*60+$5.14*120+$5.7*120 = $1,540.8 USD
	- [ ] Exclusion criteria for all rounds, after collection:
		- [ ] Turkers who choose agrees/disagrees on screen Q for which answer is disagrees/agrees
		- [ ] For each Turker, calculate %items for which all other 5 Turkers chose agrees/disagrees but they chose disagrees/agrees; if this % is greater than N--exclude. N>=10%?
		- [ ] If find an effect from party--drop annotations s.t. balance of annotations per item from D and R
	- increase task size, check IRR b/w first and second half of exp (60)
	- **looking at mean will end up being function of how many D vs. R raters--be careful


   - [ ] Round 1: 300 sentences annotated; 6 annotators per (need 60 annotators total)
	- [ ] create "gold" labels for subsample? filter Turkers based on X% accuracy?
        - [ ] within-annotator consistency
        - [ ] 4-5 rounds w/ same set of sanity questions per round
   - [ ] How to invite specific annotators?
   - [ ] How to limit number of HITs per worker?
   - [ ] Analyze annotator's demographics--may have effect, especially on more ambiguous sentences


- Miscellaneous ideas:
	- Track whether people decide to click for more context--signal of the ambiguity of the sentence (medium priority)
	- Question mark button for more information about abbreviations? (Can we track whether someone's clicked on a question mark?)
	- Future study in which I manipulate stance-taking verb
            - How do you mix them? Across, within speaker thing?

References:
   - https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde
   - http://www.saifmohammad.com/WebDocs/StarSem2016-stance-tweets.pdf

