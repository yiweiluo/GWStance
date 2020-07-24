# Climate change stance detection

`run_weighted.py` and `run_bert.py` adapt `run_glue.py` from the Transformers repository and apply our weighted and non-weighted models for classification, respectively. 

Sample usage:
```
export MODEL_TYPE=bert
export PRED_FILE_NAME=
export MODEL_NAME_OR_PATH=

python run_weighted.py \
	--model_type $MODEL_TYPE \
	--pred_file_name $PRED_FILE_NAME \
	--task_name climate-weight \
	--data_dir \
	--do_eval \
	--eval_partition pred \ 			# set to `pred` so that the model makes predictions
	--model_name_or_path $MODEL_NAME_OR_PATH \ 	
	--output_dir \
	--do_prediction \ 				# include so that model makes predictions
```