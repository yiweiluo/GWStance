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
   - [ ] Round 1: 300 sentences annotated; 6 annotators per (need 60 annotators total)
	- [ ] create "gold" labels for subsample? filter Turkers based on X% accuracy?
        - [ ] within-annotator consistency
        - [ ] 4-5 rounds w/ same set of sanity questions per round
   - [ ] How to invite specific annotators?
   - [ ] How to limit number of HITs per worker?
   - [ ] Exclusion criteria:
	- [ ] more than N% disagree from mode response
	- [ ] perfectly random behavior when data is not balanced
	- increase task size, check IRR b/w first and second half of exp (60)
	- **looking at mean will end up being function of how many D vs. R raters--be careful
	- 
   - [ ] analyze annotator's demographics--may have effect, especially on more ambiguous sentences


- Miscellaneous ideas:
	- Track whether people decide to click for more context--signal of the ambiguity of the sentence (medium priority)
	- Question mark button for more information about abbreviations? (Can we track whether someone's clicked on a question mark?)
	- Future study in which I manipulate stance-taking verb
            - How do you mix them? Across, within speaker thing?

References:
   - https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde

