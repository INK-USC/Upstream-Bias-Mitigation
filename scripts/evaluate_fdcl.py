import importlib
from utils.metric_utils import *
import argparse
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TRAIN_ITER_PER_EPOCH=1288

def get_fairness(results_fdcl):
    results_fdcl = filter_na(results_fdcl)
    results_fdcl = results_fdcl.rename_axis(["method", "lr", "seed", 'ts'], axis=0)
    results_fdcl[('fdcl_test','with','fprd')] = (results_fdcl[('fdcl_test','with','fpr_overall')] - results_fdcl[('fdcl_test', 'full', 'fpr_overall')])
    results_fdcl[('fdcl_test','with','fnrd')] = (results_fdcl[('fdcl_test','with','recall_overall')] - results_fdcl[('fdcl_test', 'full', 'recall_overall')])
    results_fdcl[('fdcl_dev','with','fprd')] = (results_fdcl[('fdcl_dev','with','fpr_overall')] - results_fdcl[('fdcl_dev', 'full', 'fpr_overall')])
    results_fdcl[('fdcl_dev','with','fnrd')] = (results_fdcl[('fdcl_dev','with','recall_overall')] - results_fdcl[('fdcl_dev', 'full', 'recall_overall')])
    results_fdcl[('fdcl_test','with','fnrd_abusive')] = (results_fdcl[('fdcl_test','with','recall_abusive')] - results_fdcl[('fdcl_test', 'full', 'recall_abusive')])
    results_fdcl[('fdcl_test','with','fprd_abusive')] = (results_fdcl[('fdcl_test','with','fpr_abusive')] - results_fdcl[('fdcl_test', 'full', 'fpr_abusive')])
    results_fdcl[('fdcl_test','with','fnrd_hateful')] = (results_fdcl[('fdcl_test','with','recall_hateful')] - results_fdcl[('fdcl_test', 'full', 'recall_hateful')])
    results_fdcl[('fdcl_test','with','fprd_hateful')] = (results_fdcl[('fdcl_test','with','fpr_hateful')] - results_fdcl[('fdcl_test', 'full', 'fpr_hateful')])
    cnt = results_fdcl.groupby(level=[0,1]).count()
    return results_fdcl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='runs/fdcl')
    args = parser.parse_args()
    df_fdcl = read_data(base='runs/fdcl', task='fdcl', name='vanilla', seed=1, lr='1e-5', split='test', header=None,
                        full_name=True, path='runs/fdcl/roberta_fdcl_vanilla_1e-5/1/predictions_fdcl_dev.csv',
                        test_task='fdcl')
    aae_fdcl, sae_fdcl, _ = partition_by_aae_given_task_split(task='fdcl', split='test', thres=0.1)
    df_fdcl_dev = read_data(base='runs/fdcl', task='fdcl', name='vanilla', seed=1, lr='1e-5', split='dev', header=None,
                            full_name=True, path='runs/fdcl/roberta_fdcl_vanilla_1e-5/1/predictions_fdcl_dev.csv',
                            test_task='fdcl')
    aae_fdcl_dev, sae_fdcl_dev, _ = partition_by_aae_given_task_split(task='fdcl', split='dev', thres=0.1)

    results_fdcl = aggregate_fdcl_results(
                                          [_ for _ in get_model_names(args.base) if 'subset' not in _],
                                          ['1e-5','5e-6','2e-5'],
                                            aae_fdcl,
                                            sae_fdcl,
                                            base=args.base,
                                            full_name=True,
                                            max_seed=3,
                                            all_checkpoints=True,
                                            w_idx_dev=aae_fdcl_dev, wo_idx_dev=sae_fdcl_dev, wwo_dataset_dev='fdcl_dev',
                                            pred_thres='equal'
                                         )

    results_fdcl = get_fairness(results_fdcl)
    best_acc_step = decide_checkpoints_with_early_stopping(results_fdcl, ('fdcl_dev', 'full', 'acc'),
                                                           by_lower=False, iter_per_epoch=TRAIN_ITER_PER_EPOCH) # number of iter per epoch in fdcl
    best_acc_step_epoch = extract_epoch_step(best_acc_step, TRAIN_ITER_PER_EPOCH)

    tab = mean_by_seed(results_fdcl.loc[
                     best_acc_step, [('fdcl_dev', 'full', 'fpr_overall'), ('fdcl_dev', 'full', 'recall_overall'),
                                     ('fdcl_test', 'with', 'fprd'), ('fdcl_dev', 'with', 'fprd'),
                                     ('brod_test', 'full', 'acc')]])
    print(tab)
    tab.to_csv(os.path.join(args.base,'results.csv'))
