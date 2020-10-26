import os
import json
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict

import numpy as np


# Train a new model using a specific config file and evaluate on test data
# config.json is the config file for a specific run, independent of folds e.g. results/<model>/config.jon

def main():
    usage = "%prog config.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('--max-epochs', type=int, default=None,
    #                  help='Number of chains: default=%default')

    (options, args) = parser.parse_args()

    config_file = args[0]
    #max_epochs = options.max_epochs

    with open(config_file) as f:
        config = json.load(f)

    basedir = config['basedir']
    transformers_dir = config['transformers_dir']
    script = config['script']
    task = config['task']
    seed = config['seed']
    model_name_or_path = config['model_name_or_path']
    base_data_dir = config['base_data_dir']
    base_outdir = config['base_outdir']
    folds = config['folds']
    lr = config['lr']
    max_seq_length = config['max_seq_length']
    weights = config['weights']
    do_text_b = config['do_text_b']
    save_steps = config['save_steps']
    max_epochs = config['max_epochs']

    all_dev_accs = []
    all_test_accs = []
    dev_acc_sums = None

    for fold in range(folds):

        dev_accs = []
        test_accs = []
        for epoch in range(max_epochs):
            if epoch == max_epochs - 1:
                infile = os.path.join(base_outdir, str(fold), 'eval_results_dev_.txt')
            else:
                infile = os.path.join(base_outdir, str(fold), 'eval_results_dev_checkpoint-' + str(int((epoch+1) * save_steps)) + '.txt')
            with open(infile) as f:
                lines = f.readlines()
            acc = float(lines[0].strip().split()[2])
            dev_accs.append(acc)

        if dev_acc_sums is None:
            dev_acc_sums = np.array(dev_accs)
        else:
            dev_acc_sums += np.array(dev_accs)
        all_dev_accs.append(dev_accs)

    
    best_epoch = int(np.argmax(dev_acc_sums))
    print("Best epoch:", best_epoch+1)
    print("Dev/test accs from training:")
    for i in range(len(all_dev_accs)):
        print(all_dev_accs[i][best_epoch])


    # Now retrain on all non-test data
    fold = 'no-dev'
    cmd = ['python', script,
           '--model_type', 'bert',
           '--model_name_or_path', model_name_or_path,
           '--task_name', task,
           '--do_train',
           '--do_lower_case',
           '--data_dir', os.path.join(base_data_dir, str(fold)),
           '--max_seq_length', str(max_seq_length),
           '--per_gpu_eval_batch_size=16',
           '--per_gpu_train_batch_size=16',
           '--learning_rate', str(lr),
           '--num_train_epochs', str(max_epochs),
           '--output_dir', os.path.join(base_outdir, str(fold)),
           '--overwrite_cache',
           '--overwrite_output_dir',
           '--seed', str(seed),
           '--num_labels', '3'
           ]

    if do_text_b:
        cmd.append('--do_text_b')

    print(cmd)
    call(cmd)

    # and then evaluate on the test data
    cmd = ['python', script,
        '--model_type', 'bert',
        '--task_name', task,
        '--do_eval',
        '--pred_file_name', "test_preds",
        '--do_lower_case',
        '--data_dir', os.path.join(base_data_dir, str(fold)),
        '--max_seq_length', str(max_seq_length),
        '--per_gpu_eval_batch_size=16',
        '--per_gpu_train_batch_size=16',
        '--output_dir',  os.path.join(base_outdir, str(fold)),
        '--overwrite_cache',
        '--overwrite_output_dir',
        '--model_name_or_path',  os.path.join(base_outdir, str(fold)),
        '--eval_partition', 'test',
        '--eval_all_checkpoints',
        '--num_labels', '3'
        ]

    if do_text_b:
        cmd.append('--do_text_b')

    print(cmd)
    call(cmd)


if __name__ == '__main__':
    main()
