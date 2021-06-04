import argparse
import copy
import os
import pandas as pd
from misc import sendmail

class SafeDict(dict):
    def __missing__(self, key):
         return '{' + key + '}'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', nargs='*', default=[1e-5])
parser.add_argument('--seed', nargs='*', default=[0])
parser.add_argument('--base', default='runs-0717')

parser.add_argument('--source')
parser.add_argument('--freeze', action='store_true')

parser.add_argument('--extra_args', nargs='?', default='')

parser.add_argument('--crit', default='f1')
parser.add_argument('--es', default=3)
parser.add_argument('--epoch', default=1000)
parser.add_argument('--subset', nargs='?')

parser.add_argument('--source_name')
parser.add_argument('--name')
parser.add_argument('--task')
parser.add_argument('--do_reg', action='store_true')
parser.add_argument('--test_task', nargs='?')

parser.add_argument('--eval', action='store_true')
parser.add_argument('--split', default='test')

parser.add_argument('--load_epoch_conf')

parser.add_argument('--emailme')

parser.add_argument('--eval_nyt', action='store_true')
parser.add_argument('--eval_brod', action='store_true')
parser.add_argument('--eval_iptts', action='store_true')

parser.add_argument('--fp32', action='store_true')

args = parser.parse_args()

if args.eval_nyt:
    args.test_task = 'nyt'
    args.eval = True
    args.split = 'test'
    args.subset = None
    args.load_epoch_conf = None
    args.extra_args += ' ##load_epoch best_f1 '

if args.eval_brod:
    args.test_task = 'brod'
    args.eval = True
    args.split = 'test'
    if args.task == 'hate':
        args.extra_args += ' ##label_num 2 '
    if args.task == 'dwmw':
        args.extra_args += ' ##label_num 3 '
    args.subset = None

if args.eval_iptts:
    args.test_task = 'iptts'
    args.eval = True
    args.split == 'test'
    args.subset = None
    args.load_epoch_conf = None
    args.extra_args += ' ##load_epoch best_f1 '



path_tmpl = '{base}/{source_name}_{task}_{name}_{lr}/{seed}'

test_task = args.task if not args.test_task else args.test_task

if test_task in ['hate', 'sentiment','offensive']:
    task_prefix = 'twitter_'
else:
    task_prefix = ''


if args.eval:
    action = '--do_eval --verbose_test'
    if args.split == 'test':
        action += ' --test'
else:
    action = '--do_train --do_eval'


if test_task in ['sentiment']:
    data_dir = 'datasets/twitter_classification/' + test_task
elif test_task in ['hate','offensive']:
    data_dir = 'datasets/twitter_classification/TweetEval/' + test_task
elif test_task in ['gab', 'nyt']:
    data_dir = 'datasets/' + test_task
elif test_task in ['fdcl']:
    data_dir = 'datasets/FDCL18'
elif test_task in ['dwmw']:
    data_dir = 'datasets/DWMW17'
elif test_task in ['brod']:
    data_dir = 'datasets/BROD16'
elif test_task in ['ws']:
    data_dir = 'datasets/ws'
elif test_task in ['toxic']:
    data_dir = 'datasets/toxic'
elif test_task in ['iptts']:
    data_dir = 'datasets/iptts77k'
elif test_task in ['emoji']:
    data_dir = 'datasets/emoji_gender'
elif test_task in ['biasbios']:
    data_dir = 'datasets/biasbios'
else:
    raise ValueError


if args.subset:
    data_dir = os.path.join(data_dir, 'subset_{}'.format(args.subset))

    
static_args = {
    'base': args.base,
    'extra_args': args.extra_args.replace('#','-') if args.extra_args else '',
    'task': args.task,
    'test_task': test_task,
    'name': args.name,
    #'dataset_base': 'TweetEval/' if args.task != 'sentiment' else '/',
    'data_dir': data_dir,
    'task_prefix': task_prefix,
    'action': action,
    'source_name': args.source_name,
    'crit': args.crit,
    'es': args.es,
    'epoch': args.epoch
}

var_args = {}
for k, v in args.__dict__.items():
    if k not in static_args:
        var_args[k] = v


        
tmpl =  "python run_model.py \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --early_stop {es} \
    --early_stop_criterion {crit} \
    --save_steps 100000 \
    --seed {seed} \
    --learning_rate {lr} \
    --num_train_epochs {epoch} \
    --output_dir={path_tmpl} \
    --model_type=roberta \
    --model_name_or_path={resume_dir} \
    --load_epoch={resume_epoch} \
    --data_dir={data_dir} \
    --evaluate_during_training \
    {action} \
    --max_seq_length=128 \
    --fp16 \
    --save_total_limit=10 \
    --task_name {task_prefix}{test_task} \
    {extra_args} ".format_map(SafeDict(path_tmpl=path_tmpl)) 

if args.fp32:
    tmpl = tmpl.replace('--fp16','')

if args.do_reg:
    tmpl += "--reg_explanations \
        --lm_dir \
        runs-0711/twitter_sentiment \
        --nb_range 0 \
        --sample_n 1 "

if not args.eval:
    tmpl += '--transfer '
    
if args.freeze:
    tmpl +=  '--freeze '

# check consistency
#if not args.do_reg and 'reg_strength' in args.extra_args:
#    raise ValueError

load_epoch_df = None
if args.load_epoch_conf:
    load_epoch_df = pd.read_pickle(args.load_epoch_conf)

for lr in var_args['lr']:
    for seed in var_args['seed']:

        # decide resume dir
        if args.eval:
            model_name_or_path = path_tmpl
        else:
            #if not args.load_epoch_conf:
            # load the default checkpoint
            model_name_or_path = args.source + '/{seed}'

        resume_epoch = -1
        if load_epoch_df is not None:
            if args.eval:
                l = '{}_{}_{}_{}'.format(args.source_name,args.task, args.name, lr).split('_')
            
            else:
                l = args.source.split('_')
            load_model_name = '_'.join(l[:-1]).split('/')[-1]
            load_lr = l[-1]
            #print(load_epoch_df)
            resume_epoch = load_epoch_df.loc[(load_model_name, load_lr, int(seed)), 'epoch']
        
        params = {
            'lr': lr,
            'seed': seed,
            'resume_dir': model_name_or_path.format_map(SafeDict(seed=seed)),
            'resume_epoch': resume_epoch
        }
        params.update(static_args)
        cmd1 = tmpl.format(**params)
        cmd = cmd1.format(**params)
        print('>>>>>\n' + cmd + '\n<<<<<')
        os.system(cmd)

if args.emailme:
    sendmail(content=args.emailme)