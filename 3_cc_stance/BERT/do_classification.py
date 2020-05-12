import os
import glob
import pandas as pd
import pickle
from collections import defaultdict
import logging
import pickle
from tqdm import tqdm
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AdamW, WEIGHTS_NAME

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import argparse
import scipy
import sklearn
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

CUDA = (torch.cuda.device_count() > 0)
CLASSES = ['for','against','neutral']
NUM_LABELS = 3

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "micro_f1": micro_f1,
        "macro_f1":macro_f1,
        "acc_and_macro_f1": (acc + macro_f1) / 2,
    }

def cm(preds, labels):
    return confusion_matrix(preds, labels)

def get_pred_label(res_,to_str=False):
    if to_str:
        return CLASSES[res_.index(max(res_))]
    else:
        return res_.index(max(res_))


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_pretrained(model, save_directory):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
    """
    assert os.path.isdir(
        save_directory
    ), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model itself if we are using distributed training
    model_to_save = model.module if hasattr(model, "module") else model

    # Attach architecture to the config
    model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # Save configuration file
    model_to_save.config.save_pretrained(save_directory)
    # with open(os.path.join(save_directory,'config.json'), 'w') as outfile:
    #     json.dump(model_to_save.config, outfile)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model weights saved in {}".format(output_model_file))


def build_dataloader(*args, sampler='random'):
    #print(args[:2])
    data = (torch.tensor(x) for x in args)
    #print(data[0])
    data = TensorDataset(*data)

    sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=1)

    return dataloader

def get_out_data(dat_path,max_seq_length=500):
    #eval_set = 'train' # can also be 'test'
    data = pd.read_csv(dat_path,
                              sep='\t',header=0,index_col=0)
    data = data.head(100)
    #data.columns = ['text']#,'outlet']

    out = defaultdict(list)

    print('Number of examples:',len(data))
    to_predict = data.clean_quote.values

    for dat_ix in range(len(data)):
        sent = to_predict[dat_ix]
        #print(sent)
        encoded_sent = tokenizer.encode(sent,add_special_tokens=True)[1:] # remove [CLS] auto-inserted at beginning
        #print('encoded sent:',encoded_sent)
        #CLS_ix = encoded_sent.index(101)
        #SEP_ix = encoded_sent[CLS_ix:].index(102)+CLS_ix
        out['input_ids'].append(encoded_sent)
        out['sentences'].append(sent)
        out['url_guid'].append(data['guid'].values[dat_ix])
        out['sent_no'].append(data['sent_no'].values[dat_ix])
        #out['index_CLS'].append(CLS_ix)
        #out['index_SEP_after_CLS'].append(SEP_ix)
        #print(encoded_sent[CLS_ix])

    out['input_ids'] = pad_sequences(
            out['input_ids'],
            maxlen=max_seq_length,
            dtype="long",
            value=0,
            truncating="post",
            padding="post")


    print('Adding attention masks...')
    # get attn masks
    for sent_no,sent in enumerate(out['input_ids']):
        tok_type_ids = [0 for tok_id in sent]
        #print('tok type ids:',tok_type_ids
        #     )
        mask = [int(tok_id > 0) for tok_id in sent]
        #print('old mask:',mask)
        #print('CLS index:',out['index_CLS'][sent_no])
        #print('SEP index:',out['index_SEP_after_CLS'][sent_no])
        # mask = [0 if n < out['index_CLS'][sent_no] or n > out['index_SEP_after_CLS'][sent_no] else 1
        #         for n,tok_id in enumerate(sent)]
        out['attention_mask'].append(mask)
        out['token_type_ids'].append(tok_type_ids)
    #print(len(out['labels']))
    #print(sum(out['labels']))

    print('Preparing input examples for prediction...')

    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="max seq len"
    )
    parser.add_argument(
        "--output_dir",
        default='output_dir',
        type=str,
        help="output dir"
    )
    parser.add_argument(
        "--num_epochs",
        default=7,
        type=int,
        help="fine tuning epochs"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="fine tuning epochs"
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=int,
        help="fine tuning epochs"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="fine tuning epochs"
    )
    parser.add_argument(
        "--downsample",
        default=0.0,
        type=float,
        help="p = prop examples to throw out"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="where to load data from"
    )
    parser.add_argument(
        "--data_name",
        default=None,
        type=str,
        help="type of data"
    )
    parser.add_argument(
        "--base_model",
        default=None,
        type=str,
        help="base model"
    )
    parser.add_argument(
        "--casing",
        default=None,
        type=str,
        help="uncased vs cased"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="/path/to/model"
    )
    parser.add_argument(
        "--eval_on_test",
        action="store_true",
        help="whether to eval on test set"
    )
    parser.add_argument("--pred_file_name", default='preds', help="Whether to run eval on the dev set (False) or test set (True).")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    ARGS = parser.parse_args()

    if (
            os.path.exists(ARGS.output_dir)
            and os.listdir(ARGS.output_dir)
            and ARGS.do_train
            and not ARGS.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                ARGS.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if ARGS.local_rank == -1 or ARGS.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not ARGS.no_cuda else "cpu")
        ARGS.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(ARGS.local_rank)
        device = torch.device("cuda", ARGS.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        ARGS.n_gpu = 1
    ARGS.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if ARGS.local_rank in [-1, 0] else logging.WARN,
    )

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

    #writer = SummaryWriter(ARGS.output_dir + '/events

    DATA_NAME = ARGS.data_name
    BASE_MOD = ARGS.base_model
    CASING = ARGS.casing
    DATA_DIR = ARGS.data_dir#os.path.join('../data_creation/scripts/save',DATA_NAME)
    print(DATA_DIR)

    #model_path = os.path.join(PRETRAINED_MODELS_DIR,DATA_NAME,BASE_MOD,
    #                         CASING)
    model_path = ARGS.model_name_or_path
    print(model_path)

    # Load model
    config = BertConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                              config=config)

    optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate, eps=1e-8)
    model.to(ARGS.device)

    logger.info("Training/evaluation parameters %s", ARGS)

    # Load data for prediction/eval
    eval_set = 'test' if ARGS.eval_on_test else 'dev' # can also be 'test'
    eval_dat_path = os.path.join(DATA_DIR,eval_set+'.tsv')
    eval_data = get_out_data(eval_dat_path,max_seq_length=500)

    test_inputs, test_masks, test_src_guids, test_src_sent_nos = eval_data['input_ids'], eval_data['attention_mask'], eval_data['url_guid'], eval_data['sent_no']
    test_dataloader = build_dataloader(
        test_inputs, test_masks,
        sampler='order')

    # ========================================
    #               Prediction
    # ========================================
    print("")
    print("Doing predictions...")

    t0 = time.time()
    model.eval()
    losses = []
    all_preds = []
    #all_labels = []
    log = open(ARGS.output_dir + '/log', 'w')
    for step, batch in enumerate(test_dataloader):

        if CUDA:
            batch = (x.cuda() for x in batch)
        input_ids, masks = batch

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=masks)
        #print("Outputs:",outputs)
        #print("Outputs[0]:",outputs[0])
        logits = outputs[0]
        #print("Logits:",logits)
        #loss, logits = outputs

        #losses.append(loss.item())

        input_ids = input_ids.cpu().numpy()
        preds = scipy.special.softmax(logits.cpu().numpy(), axis=1)
        input_toks = [
            tokenizer.convert_ids_to_tokens(s) for s in input_ids
        ]

        for seq, pred in zip(input_toks, preds):
            log.write('+' * 40 + '\n')
            log.write(' '.join(seq) + '\n')
            log.write('pred: ' + str(np.argmax(pred)) + '\n')
            log.write('dist: ' + str(pred) + '\n')
            log.write('\n\n')

            all_preds += [pred]

    log.close()
    all_preds = np.array(all_preds)

    #avg_loss = np.mean(losses)
    #f1 = sklearn.metrics.f1_score(all_labels, np.argmax(all_preds, axis=1),average='macro')
    #acc = sklearn.metrics.accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    #auc = sklearn.metrics.roc_auc_score(all_labels, all_preds[:, 1])

    #writer.add_scalar('eval/acc', acc, epoch_i)
    #writer.add_scalar('eval/auc', auc, epoch_i)
    #writer.add_scalar('eval/f1', f1, epoch_i)
    #writer.add_scalar('eval/loss', f1, epoch_i)

    #print("  Loss: {0:.2f}".format(avg_loss))
    #print("  Accuracy: {0:.2f}".format(acc))
    #print("  F1: {0:.2f}".format(f1))
    #print("  AUC: {0:.2f}".format(auc))
    print("  Evaluation took: {:}".format(format_time(time.time() - t0)))

    # Want to save: model, config, vocab, eval results, preds, training_args,
    # result = {'acc': acc,
    #     'f1':f1,
    #     'cm':cm(np.argmax(all_preds, axis=1),all_labels)}

    preds_df = pd.DataFrame({'src_guid':test_src_guids,
    'src_sent_no':test_src_sent_nos,
    'predicted':np.argmax(all_preds, axis=1)})
    preds_df.to_csv(ARGS.output_dir+'/{}.tsv'.format(ARGS.pred_file_name),sep='\t',index=False)

    # output_eval_file = os.path.join(ARGS.output_dir, "eval_results_{}.txt".format(ARGS.pred_file_name))
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
