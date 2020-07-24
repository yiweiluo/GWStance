# Climate change stance detection

`run_weighted.py` and `run_bert.py` adapt `run_glue.py` from the Transformers repository and apply our weighted and non-weighted models for classification, respectively. 

Sample usage:
```
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=/u/scr/yiweil/sci_debates/cc_stance/2span/base_s787846414_lr1e-05_msl256_2span_weights/no-dev

export PRED_FILE_NAME=batch_0_pred
export BATCH_NO=0
export DATA_DIR=/u/scr/yiweil/sci_debates/cc_stance/curr_comp_clauses/$BATCH_NO

python run_weighted.py \
	--model_type $MODEL_TYPE \
	--pred_file_name $PRED_FILE_NAME \
	--task_name climate-weight \
	--data_dir $DATA_DIR \
	--do_eval \
	--eval_partition pred \ 			# set to `pred` so that the model makes predictions
	--model_name_or_path $MODEL_NAME_OR_PATH \ 	
	--output_dir /u/scr/yiweil/sci_debates/cc_stance/curr_comp_clauses/output \
	--do_prediction \ 				# include so that model makes predictions
	--do_text_b					# whether to do 1span or 2span prediction
```