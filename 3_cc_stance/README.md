# Climate change stance detection

`1_MTurk` contains the annotations (`full_annotations.tsv`) and subject demographic info (`full_subj_info.tsv`) that we obtain from 5 rounds of data collection through Amazon Mechanical Turk. Additional file details:

* `full_annotations.tsv` has the following fields for each item: 
  * `sentence` (text shown to annotators), 
  * `worker_n`, where `n` ranges from 0-7 (annotation from the nth worker)
  * `round` (round ID, one of 1-5), 
  * `batch` (batch number, ranging from 0-9), 
  * `sent_id` (ID of item within each batch. IDs beginning with 's' indicate a screen item; IDs beginning with 't' indicate a true item),
  * `av_rating` (average rating over all 8 workers, with the label mapping `{"agree": 1, "neutral": 0, "disagree": -1}`,
  * `disagree`, `agree`, `neutral` (respective probabilities that the true label is the one given by the column name, as estimated by our item-response model),
  * `MACE_pred` (true label as estimated by [MACE](https://github.com/dirkhovy/mace))

* `full_subj_info.tsv` has the following fields for each annotator:
  * `startDate` (date of annotation)
  * `startTime` (time at which HIT was begun)
  * `endTime` (time at which HIT was completed)
  * `timeSpent` (total time spent on HIT)
  * `comments` (free-form general feedback on HIT)
  * `criticisms` (free-form feedback on aspects of HIT to be improved)
  * `round` (round ID, one of 1-5), 
  * `batch` (batch number, ranging from 0-9), 
  * `HitCorrect` (self-reported response to whether they think they did the HIT correctly)
  * `HitFamiliar` (self-reported response to whether they think they had done the HIT before)
  * `age`, `gender`, `education`, `party`, `state` (self-reported age, gender, level of education, political affiliation, and state of residence)
  * `poll*` (responses to each of 7 total poll questions that gauge stance toward climate change, see `1_MTurk/poll_questions.txt` for the full questions and answer choices) 

