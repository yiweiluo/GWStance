import os
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser


# Run a BERT model on all folds with specified options

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--folds', type=int, default=5,
                      help='Number of test folds: default=%default')
    parser.add_option('--start_fold', type=int, default=0,
                      help='Fold to start training at: default=%default')
    parser.add_option('--model_name_or_path', type=str, default='bert-base-uncased',
                      help='Path to model: default=%default')
    parser.add_option('--lr', type=float, default=2e-5,
                      help='Learning rate: default=%default')
    parser.add_option('--max_seq_length', type=int, default=256,
                      help='max seq length: default=%default')
    parser.add_option('--do_text_b', action="store_true", default=False,
                      help='Use second sequence: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--epochs', type=int, default=7,
                      help='Number of epochs: default=%default')
    parser.add_option('--save-steps', type=int, default=94,
                      help='Steps between each save: default=%default')
    parser.add_option('--basedir', type=str, default='splits/',
                      help='Output directory from split_data.py: default=%default')
    parser.add_option('--exp-dir', type=str, default='results',
                      help='Experiment output directory: default=%default')
    parser.add_option('--transformers-dir', type=str, default='../transformers/',
                      help='Transformers dir: default=%default')
    parser.add_option('--weights', action="store_true", default=False,
                      help='Output the version with weights for each possible label: default=%default')
    parser.add_option('--device', type=int, default=None,
                      help='CUDA device: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    transformers_dir = options.transformers_dir
    seed = options.seed
    epochs = options.epochs
    model_name_or_path = options.model_name_or_path
    folds = options.folds
    start_fold = options.start_fold
    lr = options.lr
    max_seq_length = options.max_seq_length
    do_text_b = options.do_text_b
    weights = options.weights
    save_steps = options.save_steps
    exp_dir = options.exp_dir
    device = options.device

    run_folds(basedir, transformers_dir, seed, model_name_or_path, folds, start_fold, lr, max_seq_length, epochs, do_text_b, weights, save_steps, exp_dir, device)


def run_folds(basedir, transformers_dir, seed=42, model_name_or_path='bert-base-uncased', folds=5, start_fold=0, lr=2e-5, max_seq_length=256, epochs=2, do_text_b=False, weights=False, save_steps=93, exp_dir='results', device=0):

    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if weights:
        task = 'climate-weight'
        subdir = 'folds'
        script = os.path.join(transformers_dir, 'run_weighted.py')
        save_steps *= 3
    else:
        task = 'climate'
        script = os.path.join(transformers_dir, 'run.py')
        subdir = os.path.join('folds')

    base_data_dir = os.path.join(basedir, task, subdir)

    if model_name_or_path == 'bert-base-uncased':
        output_prefix = 'base'
    else:
        output_prefix = 'pretrained'
    output_prefix += '_s' + str(seed)
    output_prefix += '_lr' + str(lr)
    output_prefix += '_msl' + str(max_seq_length)
    if do_text_b:
        output_prefix += '_2span'
    else:
        output_prefix += '_1span'
    if weights:
        output_prefix += '_weights'

    base_outdir = os.path.join(exp_dir, output_prefix)

    config = {'basedir': basedir,
              'transformers_dir': transformers_dir,
              'script': script,
              'task': task,
              'seed': int(seed),
              'model_name_or_path': model_name_or_path,
              'base_data_dir':  base_data_dir,
              'base_outdir': base_outdir,
              'folds': int(folds),
              'lr': float(lr),
              'max_seq_length': int(max_seq_length),
              'do_text_b': do_text_b,
              'weights': weights,
              'save_steps': int(save_steps),
              'max_epochs': int(epochs)
              }

    for fold in range(start_fold, folds):
        cmd = ['python', script,
            '--model_type', 'bert',
            '--model_name_or_path', model_name_or_path,
            '--task_name', task,
            '--do_train',
            '--do_eval',
            '--pred_file_name', "dev_preds",
            '--do_lower_case',
            '--data_dir', os.path.join(base_data_dir, str(fold)),
            '--max_seq_length', str(max_seq_length),
            '--per_gpu_eval_batch_size=16',
            '--per_gpu_train_batch_size=16',
            '--learning_rate', str(lr),
            '--num_train_epochs', str(epochs),
            '--output_dir', os.path.join(base_outdir, str(fold)),
            '--overwrite_cache',
            '--overwrite_output_dir',
            '--eval_all_checkpoints',
            '--save_steps', str(save_steps),
            '--seed', str(seed),
            '--num_labels', '3'
            ]

        if do_text_b:
            cmd.append('--do_text_b')

        print(cmd)

        my_env = os.environ.copy()
        if device is not None:
            print("Setting CUDA device")
            my_env["CUDA_VISIBLE_DEVICES"] = str(device)
        call(cmd, env=my_env)

        # Don't evaluate on test data yet
        """
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
        call(cmd, env=my_env)
        """

        with open(os.path.join(base_outdir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, sort_keys=False)

        # Delete checkpoint directories and model file to save space
        checkpoint_dirs = glob.glob(os.path.join(base_outdir, str(fold), 'checkpoint*'))
        for d in checkpoint_dirs:
            shutil.rmtree(d)

        model_file = os.path.join(base_outdir, str(fold), 'pytorch_model.bin')
        try:
            os.remove(model_file)
        except Exception as e:
            print("**** Error removing", model_file, '****')
            print(e)


if __name__ == '__main__':
    main()
