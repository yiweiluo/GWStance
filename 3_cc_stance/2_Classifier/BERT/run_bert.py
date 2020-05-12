import os
import glob
import shutil
import json
from subprocess import call
from optparse import OptionParser
from collections import defaultdict

def parse_eval_res(f_contents):
    d_ = json.loads(f_contents[1].strip().split(' = ')[-1].replace("\'", "\""))
    return d_


def get_best_checkpoint(checkpoints_dir,subdir,fold,by='dev'):
    """Get checkpoint with best accuracy (on dev set, by default)"""
    results = defaultdict(dict)
    checkpoints = [x.split('/')[-1] for x in glob.glob(os.path.join(checkpoints_dir,'checkpoint*'))]
    for n,checkpoint in enumerate(checkpoints):
        for eval_ in ['dev','test']:
            with open(os.path.join(checkpoints_dir,'eval_results_{}_{}.txt'.format(eval_,checkpoint))) as f:
                res_d = parse_eval_res(f.readlines())
            results[checkpoint][eval_] = res_d
    for eval_ in ['dev','test']:
        with open(os.path.join(checkpoints_dir,'eval_results_{}_.txt'.format(eval_))) as f:
            res_d = parse_eval_res(f.readlines())
        results['final_epoch'][eval_] = res_d

    #print('Checkpoint results:',results)
    #best_n = int(sorted(results.items(),key=lambda x: x[1][by]['acc'],reverse=True)[0][0].split('_')[-1])-1
    best_n = sorted(results.items(),key=lambda x: x[1][by]['acc'],reverse=True)[0][0]#.split('_')[-1])-1
    #print('Best epoch_no:',best_n)
    if best_n == 'final_epoch':
        return os.path.join(checkpoints_dir)
    else: # best performance achieved in final epoch
        return best_n


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--folds', type=int, default=10,
                      help='Number of test folds: default=%default')
    parser.add_option('--start_fold', type=int, default=0,
                      help='Fold to start training at')
    parser.add_option('--model_name_or_path', type=str, default=None,
                      help='Path to model')
    parser.add_option('--base_model', type=str, default=None,
                      help='Base model')
    parser.add_option('--max_seq_length', type=int, default=None,
                      help='max seq length')
    parser.add_option('--do_text_b', action="store_true",
                      help='Use second sequence: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--basedir', type=str, default='/u/scr/yiweil/sci_debates/cc_stance/climate_data',
                      help='Output directory from split_data.py: default=%default')
    parser.add_option('--transformers-dir', type=str, default='../transformers/',
                      help='Transformers dir: default=%default')
    parser.add_option('--datatype', type=str, default='all_mturk_train',
                      help='Transformers dir: default=%default')
    parser.add_option('--extreme', action="store_true", default=False,
                      help='Output the version using only extreme probs for agree and disagree: default=%default')
    parser.add_option('--weights', action="store_true", default=False,
                      help='Output the version with weights for each possible label (not extreme): default=%default')
    parser.add_option('--do_train', action="store_true")

    (options, args) = parser.parse_args()

    basedir = options.basedir
    transformers_dir = options.transformers_dir
    datatype = options.datatype
    seed = options.seed
    model_name_or_path = options.model_name_or_path
    base_model = options.base_model
    start_fold = options.start_fold
    max_seq_length = str(options.max_seq_length)
    do_text_b = options.do_text_b

    if options.weights:
        task = 'climate-weight'
        subdir = 'folds'
        script = os.path.join(transformers_dir, 'run_weighted.py')
    else:
        task = 'climate'
        script = os.path.join(transformers_dir, 'run.py')
        if options.extreme:
            subdir = 'extreme/folds'
        else:
            subdir = 'vanilla/folds'

    output_prefix = 'output'

    for fold in range(start_fold,options.folds):
        cmd = ['python', script,
            '--model_type', 'bert',
            '--model_name_or_path', model_name_or_path,
            '--task_name', task,
            '--do_train',
            '--do_eval',
            '--pred_file_name', "dev_preds",
            '--do_lower_case',
            '--data_dir', os.path.join(basedir, task, datatype, subdir, str(fold)),
            '--max_seq_length', max_seq_length,
            '--per_gpu_eval_batch_size=16',
            '--per_gpu_train_batch_size=16',
            '--learning_rate', '2e-5',
            '--num_train_epochs', '7.0',
            '--output_dir', os.path.join(basedir, task, datatype, subdir, str(fold), output_prefix, base_model),
            '--overwrite_cache',
            '--overwrite_output_dir',
            '--eval_all_checkpoints',
            '--save_steps', '103'
            ]


        if do_text_b:
            cmd.append('--do_text_b')

        if options.do_train:
            print(cmd)
            call(cmd)

            checkpoints_dir = os.path.join(basedir, task, datatype, subdir, str(fold), output_prefix, base_model)
            best_checkpoint = os.path.join(checkpoints_dir,get_best_checkpoint(checkpoints_dir,subdir,fold))
            print('Best checkpoint:',best_checkpoint)

            checkpoint_dirs = glob.glob(os.path.join(basedir, task, datatype, subdir, str(fold), output_prefix, base_model, 'checkpoint*'))
            #checkpoint_dirs.remove(best_checkpoint)
            for d in checkpoint_dirs:
                if d != best_checkpoint:
                    print('Will remove checkpoint:',d)
                    shutil.rmtree(d)
                else:
                    print('Keeping best checkpoint:',d)

        cmd = ['python', script,
            '--model_type', 'bert',
            '--task_name', task,
            '--do_eval',
            '--pred_file_name', "test_preds",
            '--do_lower_case',
            '--data_dir', os.path.join(basedir, task, datatype, subdir, str(fold)),
            '--max_seq_length', max_seq_length,
            '--per_gpu_eval_batch_size=16',
            '--per_gpu_train_batch_size=16',
            '--output_dir', os.path.join(basedir, task, datatype, subdir, str(fold), output_prefix, base_model),
            '--overwrite_cache',
            '--overwrite_output_dir',
            '--model_name_or_path', os.path.join(basedir, task, datatype, subdir, str(fold), output_prefix, base_model),
            '--eval_partition', 'test',
            '--eval_all_checkpoints'
            ]

        if do_text_b:
            cmd.append('--do_text_b')

        print(cmd)
        call(cmd)


if __name__ == '__main__':
    main()
