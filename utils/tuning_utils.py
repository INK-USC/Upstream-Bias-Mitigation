from .metric_utils import *
import os

def decide_checkpoints_with_early_stopping(df, key, by_lower, iter_per_epoch=None):
    def best_row(group):
        arr = group.loc[:, key]#.to_numpy().astype(np.float32)
        #print(arr)
        #print(np.isnan(arr))
        #print('---')
        if iter_per_epoch:
            steps = []
            for i, (index, row) in enumerate(arr.items()):
                if int(index[-1]) // iter_per_epoch == 0:
                    steps.append(i)
            arr = arr.iloc[steps]
        arr = arr.to_numpy().astype(np.float32)

        if len(arr[~np.isnan(arr)]) == 0:
            best_lr_idx = 0
        else:
            if by_lower:
                best_lr_idx = np.nanargmin(arr)
            else:
                best_lr_idx = np.nanargmax(arr)
        # return group.iloc[best_seed_idx]
        return group.iloc[best_lr_idx].name

    return df.groupby(level=['method','lr','seed']).apply(best_row)


def get_early_stopping_step(df, key, early_stop):
    def get_truncate_step_by_group(group):
        global global_group
        steps = []
        rows = []
        for row_idx, row in group.iterrows():
            global_group = group
            if row_idx[-1] != -1:
                steps.append(row_idx)
                rows.append(row)
        prev_best = -1e10
        prev_best_step = None
        patience = early_stop
        for step in steps:
            if group.loc[step, key] < prev_best:
                patience -= 1
            else:
                prev_best = group.loc[step, key]
                prev_best_step = step
                patience = early_stop
            if patience <= 0:
                break
        return prev_best_step
    return df.groupby(level=['method','lr','seed']).apply(get_truncate_step_by_group)

def decide_checkpoints_to_keep(base, *steps_tab):
    ckpts_to_keep = set()
    model_bases = set()
    model_base_ckpt_dirs = set()
    kept_model = set()
    for steps in steps_tab:
        model_base = steps[:3]
        model_bases.add(model_base)
        for folder in os.listdir(os.path.join('base', model_base[0], model_base[1], model_base[2])):
            if folder.startswith('epoch'):
                model_base_ckpt_dirs.add(os.path.join('base', model_base[0], model_base[1], model_base[2], folder))

