import importlib
import utils.metric_utils
importlib.reload(utils.metric_utils)
from transformers import AutoTokenizer
from utils.tuning_utils import *
import argparse
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TRAIN_ITER_PER_EPOCH=712

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='runs/gab')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    idents_gab = load_group_identifiers_for_metrics('datasets/identity_gab.csv', tokenizer=tokenizer)
    idents_gab_raw = load_group_identifiers_raw('datasets/identity_gab.csv')

    # load sample output data; required
    df_gab = read_data(base='runs/gab', task='gab', name='roberta_gab_vanilla', seed=1, lr='1e-5', split='test', full_name=True)
    df_gab_dev = read_data(base='runs/gab', task='gab', name='roberta_gab_vanilla', seed=1, lr='1e-5', split='dev', full_name=True)
    df_iptts = read_data(path='runs/gab/roberta_gab_vanilla_1e-5/1/predictions_iptts_test.csv', full_name=True)

    # example id with / without group identifier
    w_gab, wo_gab = partition_by_group_identifiers(idents_gab, df_gab)
    w_gab_dev, wo_gab_dev = partition_by_group_identifiers(idents_gab, df_gab_dev)

    # for in-domain fprd
    gab_test_ind = partition_by_each_group_identifier(idents_gab_raw, idents_gab, df_gab)
    gab_dev_ind = partition_by_each_group_identifier(idents_gab_raw, idents_gab, df_gab_dev)
    metrics_func_gab_test = get_metrics_df_agg(gab_test_ind, idents_gab_raw)
    metrics_func_gab_dev = get_metrics_df_agg(gab_dev_ind, idents_gab_raw)

    # for iptts fprd
    idents_50 = load_group_identifiers_for_metrics('datasets/identity.csv', tokenizer=tokenizer)
    idents_50_raw = load_group_identifiers_raw('datasets/identity.csv')
    iptts_test_ind = partition_by_each_group_identifier(idents_50_raw, idents_50, df_iptts)
    metrics_func_iptts = get_metrics_df_agg(iptts_test_ind, idents_50_raw)

    results_gab = aggregate_gab_results(
        [_ for _ in get_model_names(args.base)],
        ['1e-3','2e-5','1e-5','5e-6'], w_gab, wo_gab,
        full_name=True,
        base='runs-0121',
        max_seed=6,
        all_checkpoints=True,
        w_idx_dev=w_gab_dev,
        wo_idx_dev=wo_gab_dev,
        wwo_dataset_dev='gab_dev',
        pred_thres='equal',
    )

    results_gab = compute_fairness(results_gab, 'gab')
    _, bf1e_gab_eer, _, _ = get_checkpoints(results_gab, TRAIN_ITER_PER_EPOCH, save=False, save_postfix='gabw_eer', dev_set_name='gab_dev', base=args.base)  # number of iter per epoch in gab

    # given the checkpoints with best dev f1, compute iptts acc
    results_gab = aggregate_gab_results(
        [_ for _ in get_model_names(args.base)],
        ['1e-3','2e-5','1e-5','5e-6'], w_gab, wo_gab,
        full_name=True,
        base='runs-0121/gab_twice',
        max_seed=6,
        all_checkpoints=True,
        w_idx_dev=w_gab_dev,
        wo_idx_dev=wo_gab_dev,
        wwo_dataset_dev='gab_dev',
        pred_thres='equal',
        iptts=True,
        eval_step_epoch=bf1e_gab_eer,
        metrics_ood=['fprd_agg'],
        metrics_func_ood=metrics_func_iptts
    )
    results_gab = compute_fairness(results_gab, 'gab')
    bf1_gab_eer, bf1e_gab_eer, bfa_gab_eer, bfae_gab_eer = get_checkpoints(results_gab, TRAIN_ITER_PER_EPOCH, save=False, save_postfix='gabw_eer', dev_set_name='gab_dev', base=args.base)

    mean_df = mean_by_seed(results_gab.loc[bf1_gab_eer, [('gab_dev','full','f1'),('gab_test','full','f1'),
                                                         ('gab_test','with','fprd'), ('gab_dev','with','fprd'),
                                                         ('iptts_test','full','fprd_agg')]])
    print(mean_df)
    mean_df.to_csv(os.path.join(args.base, 'results.csv'))

