import argparse
import copy
import os
from misc import sendmail
import pandas as pd

class SafeDict(dict):
    def __missing__(self, key):
         return '{' + key + '}'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', nargs='*', default=[1e-4])
parser.add_argument('--seed', nargs='*', default=[0])
parser.add_argument('--base', default='runs-0717')

parser.add_argument('--extra_args', nargs='?', default='')
parser.add_argument('--name')
parser.add_argument('--task')
parser.add_argument('--do_reg', action='store_true')
parser.add_argument('--test_task', nargs='?')
parser.add_argument('--subset', nargs='?')

parser.add_argument('--crit', default='f1')
parser.add_argument('--es', default=3)
parser.add_argument('--epoch', default=1000)

parser.add_argument('--mtl', action='store_true')
parser.add_argument('--mtl_args', nargs='*')

parser.add_argument('--eval', action='store_true')
parser.add_argument('--split', default='test')

parser.add_argument('--use_head', nargs='?')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--load_epoch_conf')

parser.add_argument('--eval_nyt', action='store_true')
parser.add_argument('--eval_brod', action='store_true')
parser.add_argument('--eval_iptts', action='store_true')

parser.add_argument('--emailme')



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

path_tmpl = '{base}/roberta_{task}_{name}_{lr}/{seed}'

test_task = args.task if not args.test_task else args.test_task

if test_task in ['hate', 'sentiment','offensive']:
    task_prefix = 'twitter_'
else:
    task_prefix = ''


if args.eval:
    action = '--do_eval --verbose_test'
    if args.split == 'test':
        action += ' --test'
    model_name_or_path = path_tmpl
else:
    action = '--do_train --do_eval'
    model_name_or_path = 'roberta-base'




data_dir_dict = {
    'sentiment': 'datasets/twitter_classification/sentiment',
    'hate': 'datasets/twitter_classification/TweetEval/hate',
    'brod': 'datasets/BROD16',
    'offensive': 'datasets/twitter_classification/TweetEval/offensive',
    'gab': 'datasets/gab',
    'nyt': 'datasets/nyt',
    'fdcl': 'datasets/FDCL18',
    'ws': 'datasets/ws',
    'toxic': 'datasets/toxic',
    'iptts': 'datasets/iptts77k',
    'emoji': 'datasets/emoji_gender',
    'biasbios': 'datasets/biasbios',
    'dwmw': 'datasets/DWMW17',
}

data_dir = data_dir_dict[test_task]

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
    --model_type=roberta-base \
    --model_name_or_path={model_name_or_path} \
    --data_dir={data_dir} \
    --evaluate_during_training \
    {action} \
    --max_seq_length=128 \
    --fp16 \
    --save_total_limit=10 \
    --task_name {task_prefix}{test_task} \
    {extra_args} ".format_map(SafeDict(model_name_or_path=model_name_or_path, path_tmpl=path_tmpl)) 

if args.do_reg:
    tmpl += "--reg_explanations \
        --lm_dir \
        runs-0711/twitter_sentiment \
        --nb_range 0 \
        --sample_n 1 \
        "
    
if args.freeze:
    tmpl +=  '--freeze '

if args.mtl:
    mtl_args = []
    for s in args.mtl_args:
        s = s.strip('"')
        mtl_args.append('"{}"'.format(s))
    mtl_args_str = ' '.join(mtl_args).replace('{','[').replace('}',']')
    
    tmpl += "--mtl --mtl_args {}".format(mtl_args_str)
    
    if args.use_head is not None:
         tmpl += '--use_head {} '.format(args.use_head)

    #print(tmpl)

load_epoch_df = None
if args.load_epoch_conf:
    load_epoch_df = pd.read_pickle(args.load_epoch_conf)



for lr in var_args['lr']:
    for seed in var_args['seed']:
        params = {
            'lr': lr,
            'seed': seed,
        }

        resume_epoch = None
        if load_epoch_df is not None:
            load_model_name = 'roberta_{}_{}'.format(args.task, args.name)
            load_lr = lr
            #print(load_epoch_df)
            resume_epoch = load_epoch_df.loc[(load_model_name, load_lr, int(seed)), 'epoch']


        params.update(static_args)
        cmd = tmpl.format(**params).replace('[','{').replace(']','}')

        if resume_epoch is not None:
            cmd += ' --resume_epoch {}'.format(resume_epoch)

        print('>>>>>\n' + cmd + '\n<<<<<')
        os.system(cmd)

if args.emailme:
    sendmail(content=args.emailme)