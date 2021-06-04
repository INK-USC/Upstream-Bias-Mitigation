import importlib
import utils.metric_utils
importlib.reload(utils.metric_utils)
from transformers import AutoTokenizer
from utils.tuning_utils import *
import argparse
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TRAIN_ITER_PER_EPOCH=247

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    idents_ws = load_group_identifiers_for_metrics('datasets/identity_ws.csv', tokenizer=tokenizer)
    idents_ws_raw = load_group_identifiers_raw('datasets/identity_ws.csv')

    # load sample output data; required
    df_ws = read_data(base='runs/ws', task='ws', name='roberta_ws_vanilla', seed=1, lr='1e-5', split='test', full_name=True)
    df_ws_dev = read_data(base='runs/ws', task='ws', name='roberta_ws_vanilla', seed=1, lr='1e-5', split='dev', full_name=True)
    df_iptts = read_data(path='runs/ws/roberta_ws_vanilla_1e-5/1/predictions_iptts_test.csv', full_name=True)

    # example id with / without group identifier
    w_ws, wo_ws = partition_by_group_identifiers(idents_ws, df_ws)
    w_ws_dev, wo_ws_dev = partition_by_group_identifiers(idents_ws, df_ws_dev)

    # for in-domain fprd
    ws_test_ind = partition_by_each_group_identifier(idents_ws_raw, idents_ws, df_ws)
    ws_dev_ind = partition_by_each_group_identifier(idents_ws_raw, idents_ws, df_ws_dev)
    metrics_func_ws_test = get_metrics_df_agg(ws_test_ind, idents_ws_raw)
    metrics_func_ws_dev = get_metrics_df_agg(ws_dev_ind, idents_ws_raw)

    # for iptts fprd
    idents_50 = load_group_identifiers_for_metrics('datasets/identity.csv', tokenizer=tokenizer)
    idents_50_raw = load_group_identifiers_raw('datasets/identity.csv')
    iptts_test_ind = partition_by_each_group_identifier(idents_50_raw, idents_50, df_iptts)
    metrics_func_iptts = get_metrics_df_agg(iptts_test_ind, idents_50_raw)

    results_ws = aggregate_ws_results(
        [_ for _ in get_model_names(args.base)],
        ['1e-3','2e-5','1e-5','5e-6'], w_ws, wo_ws,
        full_name=True,
        base='runs-0121',
        max_seed=6,
        all_checkpoints=True,
        w_idx_dev=w_ws_dev,
        wo_idx_dev=wo_ws_dev,
        wwo_dataset_dev='ws_dev',
        pred_thres='equal',
    )

    results_ws = compute_fairness(results_ws, 'ws')
    _, bf1e_ws_eer, _, _ = get_checkpoints(results_ws, TRAIN_ITER_PER_EPOCH, save=False, save_postfix='wsw_eer', dev_set_name='ws_dev', base=args.base)

    # given the checkpoints with best dev f1, compute iptts acc
    results_ws = aggregate_ws_results(
        [_ for _ in get_model_names(args.base)],
        ['1e-3','2e-5','1e-5','5e-6'], w_ws, wo_ws,
        full_name=True,
        base='runs-0121/ws_twice',
        max_seed=6,
        all_checkpoints=True,
        w_idx_dev=w_ws_dev,
        wo_idx_dev=wo_ws_dev,
        wwo_dataset_dev='ws_dev',
        pred_thres='equal',
        iptts=True,
        eval_step_epoch=bf1e_ws_eer,
        metrics_ood=['fprd_agg'],
        metrics_func_ood=metrics_func_iptts
    )
    results_ws = compute_fairness(results_ws, 'ws')
    bf1_ws_eer, bf1e_ws_eer, bfa_ws_eer, bfae_ws_eer = get_checkpoints(results_ws, TRAIN_ITER_PER_EPOCH, save=False, save_postfix='wsw_eer', dev_set_name='ws_dev', base=args.base)

    mean_df = mean_by_seed(results_ws.loc[bf1_ws_eer, [('ws_dev','full','f1'),('ws_test','full','f1'),
                                                         ('ws_test','with','fprd'), ('ws_dev','with','fprd'),
                                                         ('iptts_test','full','fprd_agg')]])
    print(mean_df)
    mean_df.to_csv(os.path.join(args.base, 'results.csv'))

