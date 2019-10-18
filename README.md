# Climate change stance classifier

TODOs

- [ ] Training and test data
   - [ ] SemEval 2016 task 6 train, trial--both very small (211 for, 15 against)
   - [ ] SemEval 2016 test (subtask A)--169 tweets (123 for, 11 against, 35 none)
   - [ ] Mohammad et al. paper--data requested
   - [ ] Dallas' annotated news data
   - [ ] Yiwei's news data--needs annotation
- [ ] MTurk 
   - [ ] pilot, iterate with smaller subsample to check inter-annotator agreement
   - [ ] analyze annotator's demographics--may have effect, especially on more ambiguous sentences

Mturk specifics:
- Target: <b>People should be worried about climate change</b>
- Demographic info: Age, gender, level of education, political ideology, gauge stance w/ smth like "do you think people should be worried about climate change", rate how easy it was to judge sentences' stances
- Intersperse obvious sentences
- Miscellaneous ideas:
	- Track whether people decide to click for more context--signal of the ambiguity of the sentence (medium priority)
	- Question mark button for more information about abbreviations? (Can we track whether someone's clicked on a question mark?)

References:
   - https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde

