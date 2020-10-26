import os
import json
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict

import numpy as np


# Make predictions on other data
# config.json is the config file for a specific run, independent of folds e.g. results/<model>/config.jon
# /path/to/model/dir/ is the directory with the final trained model e.g. results/<model>/no-dev/
# <output_prefix>.tsv will be written to the model directory
# Note that it is necessary go provide dummy labels for the input data in the usual format

def main():
    usage = "%prog config.json /path/to/model/dir/"
    parser = OptionParser(usage=usage)
    parser.add_option('--data-dir', type=str, default=None,
                      help='Directory containing test.tsv: default=%default')
    parser.add_option('--input-prefix', type=str, default='test',
                      help='Alternative name for test[.tsv] file: default=%default')
    parser.add_option('--output-prefix', type=str, default='predictions',
                      help='Output prefix: default=%default')
    parser.add_option('--transformers-dir', type=str, default=None,
                      help='Path to transformers; override value in config.json: default=%default')

    (options, args) = parser.parse_args()

    config_file = args[0]
    model_path = args[1]

    data_dir = options.data_dir
    input_prefix = options.input_prefix
    transformers_dir = options.transformers_dir

    with open(config_file) as f:
        config = json.load(f)

    script = config['script']

    script_dir, script_name = os.path.split(script)
    if transformers_dir is not None:
        print("Using transformers dir:", transformers_dir)
        script = os.path.join(transformers_dir, script_name)

    task = config['task']
    max_seq_length = config['max_seq_length']
    do_text_b = config['do_text_b']

    # evaluate on external data
    cmd = ['python', script,
           '--model_type', 'bert',
           '--task_name', task,
           '--do_eval',
           '--pred_file_name', options.output_prefix,
           '--do_lower_case',
           '--data_dir', data_dir,
           '--max_seq_length', str(max_seq_length),
           '--per_gpu_eval_batch_size=16',
           '--per_gpu_train_batch_size=16',
           '--output_dir', model_path,
           '--overwrite_cache',
           '--overwrite_output_dir',
           '--model_name_or_path', model_path,
           '--eval_partition', input_prefix,
           '--num_labels', '3'
           ]

    if do_text_b:
        cmd.append('--do_text_b')

    print(cmd)
    call(cmd)


if __name__ == '__main__':
    main()
