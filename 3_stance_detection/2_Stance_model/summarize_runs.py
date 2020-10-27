import os
import json
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict

import numpy as np


def main():
    usage = "%prog base_output_dir"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    base_dir = args[0]
    # Get those models that have completed four folds
    dirs = sorted(glob.glob(os.path.join(base_dir, '*', '4')))

    full_list_of_accs = []

    for d in dirs:
        config_file = d[:-1] + 'config.json'
        with open(config_file) as f:
            config = json.load(f)

        script = config['script']
        task = config['task']
        seed = config['seed']
        model_name_or_path = config['model_name_or_path']
        base_data_dir = config['base_data_dir']
        base_outdir = config['base_outdir']
        folds = config['folds']
        lr = config['lr']
        max_seq_length = config['max_seq_length']
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
            full_list_of_accs.append(np.max(dev_accs))

        best_epoch = int(np.argmax(dev_acc_sums))
        best_dev_accs = []
        for i in range(len(all_dev_accs)):
            best_dev_accs.append(all_dev_accs[i][best_epoch])

        print('{:.4f}'.format(np.mean(best_dev_accs)), config_file, best_epoch+1)

    with open(os.path.join(base_dir, 'all_accs.json'), 'w') as f:
        json.dump(full_list_of_accs, f)


if __name__ == '__main__':
    main()
