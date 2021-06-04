import pandas as pd

import os
#from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from scipy.special import softmax
import numpy as np
from utils.misc import load_group_identifiers_for_metrics, find_neutral_word_positions_for_metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from nltk.tokenize import word_tokenize

_fdcl_label2id = {
    'normal': 0,
    'abusive': 1,
    'hateful': 2,
    'spam': 3
}

_fdcl_label2id_adv = {
    'normal': 0,
    'spam': 1,
    'abusive': 2,
    'hateful': 3
}

def partition_by_each_group_identifier(raw_idents, idents, df):
    K = 4
    assert len(idents) == K * len(raw_idents)
    indicator = np.zeros((len(df), len(raw_idents)), dtype=np.int32)
    raw_idents_order = sorted([(i,word) for i, word in enumerate(raw_idents)], key=lambda x: -len(x[-1]))
    for i in range(len(df)):
        item = df.iloc[i]
        for j, raw_ident in raw_idents_order:
            idents_to_find = idents[K * j: K * j + K]
            nw_positions = find_neutral_word_positions_for_metrics(idents_to_find, item.text)
            if nw_positions:
                indicator[i,j] = 1
                break
    indicator = pd.DataFrame(indicator, index=df.index, columns=range(len(raw_idents)))
    # joint_df = pd.concat([df, indicator], axis=1)
    return indicator

def get_gab_group_identifier_indicators(df, tokenzier, identity_gab_file='datasets/identity_gab.csv', iptts_identity_file='datasets/identity.csv'):
    K = 4
    identity_gab_raw = load_group_identifiers_raw(identity_gab_file)
    identity_iptts_raw = load_group_identifiers_raw(iptts_identity_file)

    identity_gab, identity_iptts = load_group_identifiers_for_metrics(identity_gab_file, tokenzier),\
                                   load_group_identifiers_for_metrics(iptts_identity_file, tokenzier)

    identity, identity_raw = identity_gab + identity_iptts, identity_gab_raw + identity_iptts_raw
    filter_identity, filter_identity_raw = [], []
    for i, ident in enumerate(identity_raw):
        if ident not in filter_identity_raw:
            filter_identity_raw.append(ident)
            filter_identity.extend(identity[K * i : K * (i + 1)])

    indicators = partition_by_each_group_identifier(filter_identity_raw, filter_identity, df)
    return indicators, filter_identity, filter_identity_raw


def partition_by_group_identifiers(idents, df):
    with_nw_idx, wo_nw_idx = [], []
    for i in range(len(df)):
        item = df.iloc[i]
        nw_positions = find_neutral_word_positions_for_metrics(idents, item.text)
        if not nw_positions:
            wo_nw_idx.append(i)
        else:
            with_nw_idx.append(i)
    return with_nw_idx, wo_nw_idx

def load_group_identifiers_raw(filename):
    f = open(filename)
    res = []
    for line in f.readlines():
        word = line.split('\t')[0].strip()
        #tokens = tokenizer.tokenize(' ' + word)
        res.append(word)
    return res

def extract_group_identifiers_iptts(raw_idents, df):
    all_matched = []
    for i in range(len(df)):
        item = df.iloc[i]
        #_, matched_terms = find_neutral_word_positions_for_metrics(idents, item.text)
        text_tokens = item.Text.split()
        #print(text_tokens)
        matched = None
        for ident in raw_idents:
            if ident in text_tokens:
                matched = ident
                break
        all_matched.append(matched)
    return all_matched

# def predict_aae_roberta_tokenized(text):
#     #tokens = text.replace('Ġ','').split()
#     text = tokenizer.convert_tokens_to_string(text.split()[1:-1])
#     tokens = word_tokenize(text.lower())
#     #print(tokens)
#     res = aae_predictor.predict(tokens)
#     if res is None:
#         return -1
#     return res[0] - max(res[1:])

# def compute_aae(df):
#     df['aae_score'] = df.apply(lambda row: predict_aae_roberta_tokenized(row['text']), axis=1)
#     return df

def predict_aae(text, tokenizer):
    tokens = ' '.join(tokenizer.tokenize(text)).replace('Ġ','').split()
    res = aae_predictor.predict(tokens)
    if res is None:
        return -1
    return res[0] - max(res[1:])

def compute_aae(df, tokenizer):
    df['aae_prob'] = df.apply(lambda row: predict_aae(row['text'], tokenizer), axis=1)

def partition_by_aae_given_task_split(task, split, data_dir=None, thres=0.1):
    if task == 'twitter_sentiment':
        df = pd.read_csv('datasets/twitter_classification/sentiment/aae_{}.tsv'.format(split), sep='\t')
        df = df.iloc[1:]
    if task == 'twitter_hate':
        df = pd.read_csv('datasets/twitter_classification/TweetEval/hate/aae_{}.tsv'.format(split), sep='\t')
        df = df.iloc[1:]
    elif task == 'twitter_offensive':
        df = pd.read_csv('datasets/twitter_classification/TweetEval/offensive/aae_{}.tsv'.format(split), sep='\t')
        df = df.iloc[1:]
    elif task == 'dwmw':
        df = pd.read_csv('datasets/DWMW17/{}.csv'.format(split))
    elif task == 'fdcl':
        if data_dir is not None:
            df = pd.read_csv(os.path.join(data_dir, 'aae_{}.tsv'.format(split)), sep='\t')
        else:
            df = pd.read_csv('datasets/FDCL18/aae_{}.tsv'.format(split), sep='\t')
        df = df.iloc[1:]
    ws, wos = [], []
    for i in range(len(df)):
        if df.iloc[i].aae_prob >= thres:
            ws.append(i)
        else:
            wos.append(i)
    return ws, wos, df

def false_positive_rates(df, pos_class=[1], neg_class=[0]):
    # pred = df.apply(lambda x: 1 if x.pred_pos > x.pred_neg else 0, axis=1)
    neg_df = df.loc[df['label'] == 0]
    tnr = accuracy_score(y_true=neg_df['label'], y_pred=neg_df['pred'] )
    return 1 - tnr


def false_positive_rate_func(y_true, y_pred):
    neg = y_true[y_true==0]
    pred = y_pred[y_true==0]
    tnr = accuracy_score(y_true=neg, y_pred=pred)
    return 1 - tnr

def metrics_df_ood(df):
    acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
    return {'acc': acc, 'f1': acc}

def metrics_df(df):
    pred_probs = softmax(np.stack([df['pred_neg'], df['pred_pos']],0), 1)

    acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
    fpr = false_positive_rates(df)
    try:
        auc_roc = roc_auc_score(y_true=df['label'], y_score=pred_probs[1,:])
    except ValueError:
        auc_roc = 0

    p = precision_score(y_true=df['label'], y_pred=df['pred'])

    if 1 in df['label'].tolist():
        r = recall_score(y_true=df['label'], y_pred=df['pred'])
        f1 = f1_score(y_true=df['label'], y_pred=df['pred'])
    else:
        r = 0
        f1 = 0

    # compute
    return {'f1': f1, 'acc':acc, 'precision': p, 'recall': r, 'fpr': fpr, 'auc': auc_roc}

def metrics_df_gender(df, test_df):
    # for biasbios dataset only
    acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
    assert len(df) == len(test_df)
    df['g'] = test_df['g']
    tprds = []
    for label in range(28):
        df_subset = df[df['label'] == label]
        if len(df_subset) == 0:
            print('warning: no instance for label {}'.format(label))
            continue
        df_male_subset = df_subset[df_subset['g'] == 'm']
        df_female_subset = df_subset[df_subset['g'] == 'f']

        tpr_label_male = accuracy_score(y_true=df_male_subset['label'], y_pred=df_male_subset['pred'])
        tpr_label_female = accuracy_score(y_true=df_female_subset['label'], y_pred=df_female_subset['pred'])

        tprd = tpr_label_male - tpr_label_female

        tprds.append(tprd)

    # compute root mean square
    final_tprd = (sum([x ** 2 for x in tprds]) / len(tprds)) ** 0.5
    return {
        'acc': acc,
        'tprd': final_tprd
    }

def get_metrics_df_agg(indicators, raw_idents):
    def metrics_func(df):
        base_mets = metrics_df(df)

        df_subset_w = indicators.sum(axis=1) != 0
        overall_fpr = false_positive_rates(df[df_subset_w])

        s = 0
        if len(indicators) == len(df):
            agg_fpr_diff = 0
            joint_df = pd.concat([df, indicators], axis=1)
            for i,ident in enumerate(raw_idents):
                df_subset = joint_df[joint_df.loc[:, i] == 1]
                fpr = false_positive_rates(df_subset)
                if not np.isnan(fpr):
                    agg_fpr_diff += abs(fpr - overall_fpr) * len(df_subset)
                    s += len(df_subset)
            agg_fpr_diff /= s
        else:
            agg_fpr_diff = -1
        base_mets['fprd_agg'] = agg_fpr_diff
        return base_mets
    return metrics_func


def metrics_df_emoji(df):
    acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
    return acc


def get_label_fpnr(preds, labels, pos_labels):
    bin_preds, bin_labels = [], []
    for item in preds:
        if item in pos_labels:
            bin_preds.append(1)
        else:
            bin_preds.append(0)

    for item in labels:
        if item in pos_labels:
            bin_labels.append(1)
        else:
            bin_labels.append(0)
    bin_preds, bin_labels = np.array(bin_preds), np.array(bin_labels)
    neg = bin_labels[bin_labels == 0]
    pred = bin_preds[bin_labels == 0]
    tnr = accuracy_score(y_true=neg, y_pred=pred)
    recall = recall_score(bin_labels, bin_preds)
    return 1 - tnr, 1 - recall

def get_multi_metric_func(metric_names):
    #if adv_fdcl:
    fdcl_label2id = _fdcl_label2id_adv
    #else:
    #fdcl_label2id = _fdcl_label2id

    def metrics_multi(df):
        res_dict = {}
        acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
        macro_f1 = f1_score(y_true=df['label'], y_pred=df['pred'], average='macro')
        res_dict['acc'] = acc
        res_dict['f1'] = acc
        res_dict['macro_f1'] = macro_f1
        for label in metric_names:
            assert df.columns[0] == 'text' and df.columns[2].startswith('pred')
            pred = (df['pred'] == fdcl_label2id[label]).astype('int32')
            bin_label = (df['label'] == fdcl_label2id[label]).astype('int32')

            p,r,f1, fpr = precision_score(y_true=bin_label, y_pred=pred), recall_score(y_true=bin_label, y_pred=pred), \
                        f1_score(y_true=bin_label, y_pred=pred), false_positive_rate_func(y_true=bin_label, y_pred=pred)
            res_dict['precision_{}'.format(label)] = p
            res_dict['recall_{}'.format(label)] = r
            res_dict['f1_{}'.format(label)] = f1
            res_dict['fpr_{}'.format(label)] = fpr

        pred = df['pred'].isin([fdcl_label2id['spam'], fdcl_label2id['abusive'], fdcl_label2id['hateful']]).astype('int32')
        bin_label = df['label'].isin([fdcl_label2id['spam'], fdcl_label2id['abusive'], fdcl_label2id['hateful']]).astype('int32')

        p,r,f1, fpr = precision_score(y_true=bin_label, y_pred=pred), recall_score(y_true=bin_label, y_pred=pred), \
                        f1_score(y_true=bin_label, y_pred=pred), false_positive_rate_func(y_true=bin_label, y_pred=pred)
        res_dict['precision_overall'], res_dict['recall_overall'], res_dict['f1_overall'], res_dict['fpr_overall'] = \
            p, r, f1, fpr

        return res_dict
    return metrics_multi

def dwmw_metric_func(df):
    bin_label = df['label'].isin([0,1]).astype('int32')
    bin_pred = df['pred'].isin([0,1]).astype('int32')
    acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
    macro_f1 = f1_score(y_true=df['label'], y_pred=df['pred'], average='macro')

    fpr = false_positive_rate_func(bin_label, bin_pred)
    fnr = 1 - recall_score(bin_label, bin_pred)
    return {
        'fpr': fpr,
        'fnr': fnr,
        'acc': acc,
        'macro_f1': macro_f1
    }


def get_emoji_metric_func(pos_label_ids):
    def metrics(df):
        fpr, fnr = get_label_fpnr(df['pred'], df['label'], pos_label_ids)
        acc = accuracy_score(y_true=df['label'], y_pred=df['pred'])
        macro_f1 = f1_score(y_true=df['label'], y_pred=df['pred'], average='macro')
        res_dict = {
            'fpr': fpr,
            'fnr': fnr,
            'acc': acc,
            'macro_f1': macro_f1
        }
        return res_dict
    return metrics

def bin_thres(df_dev):
    fpr, tpr, threshold = roc_curve(df_dev['label'], df_dev['pred_pos'] - df_dev['pred_neg'], pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold

def fdcl_thres(df_dev):
    fdcl_label2id = _fdcl_label2id_adv
    df_dev.columns = ['text', 'label', 'pred_none', 'pred_spam', 'pred_abusive', 'pred_hateful']
    prob = df_dev[['pred_abusive', 'pred_hateful', 'pred_spam']].max(axis=1) - df_dev[['pred_none']].max(axis=1)
    bin_label = df_dev['label'].apply(lambda x: 1 if x in [fdcl_label2id['abusive'], fdcl_label2id['hateful'], fdcl_label2id['spam']] else 0)

    fpr, tpr, threshold = roc_curve(bin_label, prob, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold

def dwmw_thres(df_dev):
    df_dev.columns = ['text', 'label', 'pred_hateful', 'pred_abusive', 'pred_none']
    prob = df_dev[['pred_hateful', 'pred_abusive']].max(axis=1) - df_dev[['pred_none']].max(axis=1)
    #pred = df['pred'].apply(lambda x: 1 if x !=neutral_class else 0)
    bin_label = df_dev['label'].apply(lambda x: 1 if x in [0,1] else 0)

    fpr, tpr, threshold = roc_curve(bin_label, prob, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold


def read_data(task=None, split='test', test_task=None, path=None, pred_thres=None, full_name=False, thres=None,
              return_thres=False,**path_tmpl_args):
    #if full_name:
    path_tmpl = '{base}/{name}_{lr}/{seed}'
    #else:
    #    path_tmpl = '{base}/roberta_{task}_{name}_{lr}/{seed}'

    if path is None:
        path = path_tmpl.format(task=task,**path_tmpl_args)
        if test_task is None:
            test_task = task
        df = pd.read_csv(os.path.join(path,'predictions_{}_{}.csv'.format(test_task, split).replace('brod2','brod').replace('brod3','brod')), header=None)
    else:
        path = path.replace('brod2','brod').replace('brod3','brod')
        #print(path)
        df = pd.read_csv(path, header=None)
    if test_task == 'fdcl':
        df.columns = ['text', 'label', 'pred_none', 'pred_spam', 'pred_abusive', 'pred_hateful']
        if pred_thres is None:
            df['pred'] = df.loc[:, ['pred_none','pred_spam', 'pred_abusive', 'pred_hateful']].to_numpy().argmax(axis=-1)
        elif pred_thres == 'equal':
            path = path.replace('_test_', '_dev_')
            df_dev = pd.read_csv(path,header=None)
            thres = fdcl_thres(df_dev)
            df['pred_none'] = df['pred_none'] + thres
            #df['pred_spam'] = df['pred_spam'] + thres

            df['pred'] = df[['pred_none', 'pred_spam', 'pred_abusive', 'pred_hateful']].to_numpy().argmax(axis=-1)
    elif test_task == 'dwmw':
        df.columns = ['text', 'label', 'pred_hateful', 'pred_abusive', 'pred_none']
        if pred_thres is None:
            #df['pred'] = df.apply(lambda x: x.iloc[2:6].to_numpy().argmax(), axis=1)
            df['pred'] = df.loc[:, ['pred_hateful', 'pred_abusive', 'pred_none']].to_numpy().argmax(axis=-1)
        elif pred_thres == 'equal':
            path = path.replace('_test_', '_dev_')
            df_dev = pd.read_csv(path,header=None)
            thres = dwmw_thres(df_dev)
            #print(thres, path)
            df['pred_none'] = df['pred_none'] + thres
            #df['pred_spam'] = df['pred_spam'] + thres

            df['pred'] = df[['pred_hateful', 'pred_abusive', 'pred_none']].to_numpy().argmax(axis=-1)
    elif test_task == 'emoji':
        df.columns = ['text', 'label'] + [str(i) for i in range(10)]
        df['pred'] = df.loc[:, [str(i) for i in range(10)]].to_numpy().argmax(axis=-1)
    elif test_task == 'biasbios':
        df.columns = ['text', 'label'] + [str(i) for i in range(28)]
        df['pred'] = df.loc[:, [str(i) for i in range(28)]].to_numpy().argmax(axis=-1)
    elif test_task == 'twitter_sentiment':
        df.columns = ['text', 'label', 'pred_negative', 'pred_neutral', 'pred_positive']
        df['pred'] = df.apply(lambda x: x.iloc[2:5].to_numpy().argmax(), axis=1)
    elif test_task == 'brod':
        df.columns = ['text', 'label', 'pred_none', 'pred_spam', 'pred_abusive', 'pred_hateful']
        if pred_thres is not None:
            df['pred_none'] = df['pred_none'] + thres
        df['pred'] = df[['pred_none', 'pred_spam', 'pred_abusive', 'pred_hateful']].to_numpy().argmax(axis=-1)
    elif test_task == 'brod2':
        df.columns = ['text','label','pred_neg','pred_pos']
        if pred_thres is not None:
            df['pred_neg'] = df['pred_neg'] + thres
        df['pred'] = df[['pred_neg', 'pred_pos']].to_numpy().argmax(axis=-1)
    elif test_task == 'brod3':
        df.columns = ['text', 'label', 'pred_hateful', 'pred_abusive', 'pred_none']
        if pred_thres is not None:
            df['pred_none'] = df['pred_none'] + thres
        df['pred'] = df[['pred_none', 'pred_hateful', 'pred_abusive']].to_numpy().argmax(axis=-1)
    elif test_task == 'nyt':
        df.columns = ['text','label','pred_neg','pred_pos']
        if pred_thres is not None:
            df['pred_neg'] = df['pred_neg'] + thres
        df['pred'] = df[['pred_neg', 'pred_pos']].to_numpy().argmax(axis=-1)
    else:
        df.columns = ['text','label','pred_neg','pred_pos']
        if pred_thres is None:
            df['pred'] = ((df['pred_pos'] - df['pred_neg']) > 0).astype('int32')
        elif pred_thres == 'equal':
            path = path.replace('_test_','_dev_')
            df_dev = pd.read_csv(path, header=None)
            df_dev.columns = df.columns
            thres = bin_thres(df_dev)
            df['pred'] = ((df['pred_pos'] - df['pred_neg']) > thres).astype('int32')

    if return_thres:
        return df, thres
    else:
        return df

def get_all_checkpoints(base, dataset):
    if os.path.isdir(base):
        files = os.listdir(base)
    else:
        return []
    checkpoint_files = []
    for file in files:
        #print(file)
        if file.startswith('predictions_{}'.format(dataset)) and file.endswith('.csv'):
            meta = file[:-4].split('_')
            epoch, global_iter = -1, -1
            if 'epoch' in meta:
                epoch = int(meta[meta.index('epoch') + 1])
            if 'global' in meta:
                global_iter = int(meta[meta.index('global') + 1])
            checkpoint_files.append((file, epoch, global_iter))
    #print(checkpoint_files)
    checkpoint_files.sort(key=lambda x: x[-1])
    return checkpoint_files

def compute_metrics_from_csv(dev_csv_path, test_csv_path, task, tokenizer, eer, data_dir=None,
                             split='dev'):
    df_dev, df_test = read_data(task=task, split='dev', test_task=task, path=dev_csv_path,
                                  pred_thres='equal' if eer else None,
                                  full_name=True), \
                        read_data(task=task, split='test', test_task=task, path=test_csv_path,
                                  pred_thres='equal' if eer else None,
                                  full_name=True) if test_csv_path else None

    if split == 'dev':
        df = df_dev
    elif split == 'test':
        df = df_test
    else:
        raise NotImplementedError(split)

    metrics = {}
    if task in ['gab', 'ws', 'hate', 'twitter_hate','twitter_offensive']:
        idents_gab = load_group_identifiers_for_metrics(tokenizer=tokenizer, filename='datasets/identity_gab.csv')
        w, wo = partition_by_group_identifiers(idents_gab, df)

        subsets = ['full', 'with', 'wo']
        subset_df = [df, df.iloc[w], df.iloc[wo]]

        for subset, df_s in zip(subsets, subset_df):
            metrics[subset] = metrics_df(df_s)

        # fpr
        metrics['fprd'] = abs(metrics['with']['fpr'] - metrics['full']['fpr'])
        metrics['f1'] = metrics['full']['f1']
    elif task in ['fdcl']:
        ws, wos, df_san = partition_by_aae_given_task_split(task, split=split, data_dir=data_dir)
        #ws_dev, wos_dev, df_san_dev = partition_by_aae_given_task_split(task, split='dev', data_dir=data_dir)
        assert len(df_san) == len(df_dev)

        subsets = ['full', 'with', 'wo']
        subset_df = [df, df.iloc[ws], df.iloc[wos]]

        metric_func = get_multi_metric_func(['spam','abusive','hateful'])

        for subset, df_s in zip(subsets, subset_df):
            metrics[subset] = metric_func(df_s)

        metrics['f1'] = metrics['full']['acc'] # acc, micro-averaged
        metrics['fprd'] = abs(metrics['full']['fpr_overall'] - metrics['with']['fpr_overall'])
        metrics['macro_f1'] = metrics['full']['macro_f1']
    elif task in ['emoji', 'biasbios']:
        metrics['f1'] = metrics['acc'] = metrics_df_emoji(df)
        metrics['fprd'] = 0
        metrics['macro_f1'] = 0
    elif task in ['dwmw']:
        metrics['f1'] = metrics['acc'] = metrics_df_emoji(df)
        metrics['fprd'] = 0
        metrics['macro_f1'] = 0
    return metrics

def aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, task, wwo_dataset, pred_thres=None, metrics_func=metrics_df,
                     full_name=False, base='runs-0711', test_task=None, max_seed=3, all_checkpoints=False,
                     wwo_dataset_dev=None, w_idx_dev=None, wo_idx_dev=None, subsets=('full', 'with', 'wo'), indices_dev=None,
                     indices_test=None, metrics_func_ood=metrics_df, metrics_ood=None, metrics_func_dev=None, eval_step_epoch=None, **kwargs):
    if metrics_func_dev is None:
        metrics_func_dev = metrics_func

    if test_task is None:
        test_task = task
    seeds = [_ for _ in range(max_seed)]

    if full_name:
        base_dir = '{}/{}_{}/{}/'.format(base, names[0], lrs[0], seeds[0])
    else:
        base_dir = '{}/roberta_{}_{}_{}/{}/'.format(base, task.split('_')[-1], names[0], lrs[0], seeds[0])
    checkpoint_files_ex = get_all_checkpoints(base_dir, datasets[0])
    checkpoint_global_iters = [_[-1] for _ in checkpoint_files_ex]


    multi_row_index = pd.MultiIndex.from_product([names, lrs, seeds]) if not all_checkpoints \
                else pd.MultiIndex.from_product([names, lrs, seeds, checkpoint_global_iters])

    multi_col_index = pd.MultiIndex.from_product([datasets, subsets, metrics])
    res_df = pd.DataFrame(index=multi_row_index, columns=multi_col_index)
    pred_thres_df = pd.DataFrame(index=multi_row_index, columns=['thres'])

    ds2len = {}

    for name in names:
        for lr in lrs:
            for seed in seeds:
                if full_name:
                    base_dir = '{}/{}_{}/{}/'.format(base, name, lr, seed)
                else:
                    base_dir = '{}/roberta_{}_{}_{}/{}/'.format(base, task.split('_')[-1], name, lr, seed)

                previous_thres = None
                prev_global_iters = None
                for dataset in datasets:
                    is_ood = dataset != wwo_dataset and dataset != wwo_dataset_dev
                    all_thres = []
                    if all_checkpoints:
                        checkpoint_files = get_all_checkpoints(base_dir, dataset)
                        paths = [(base_dir + checkpoint_file[0]) for checkpoint_file in checkpoint_files] if not is_ood \
                            else [base_dir + 'predictions_{}.csv'.format(dataset)] * len(prev_global_iters)
                        global_iters = [checkpoint_file[-1] for checkpoint_file in checkpoint_files] if not is_ood \
                            else prev_global_iters

                        prev_global_iters = global_iters
                    else:
                        paths = [base_dir + 'predictions_{}.csv'.format(dataset)]
                        global_iters = []

                    if eval_step_epoch is not None:
                        try:
                            eval_global_step = eval_step_epoch.loc[(name, lr, seed), 'step']
                            idx = global_iters.index(eval_global_step)
                            global_iters = global_iters[idx: idx + 1]
                            paths = paths[idx: idx + 1]
                            prev_global_iters = global_iters
                        except KeyError:
                            #print('step epoch not found for {}, {}, {}'.format(name, lr, seed))
                            pass

                    try:
                        for pi,(path,global_iter) in enumerate(zip(paths, global_iters)):
                            dfs = []
                            if is_ood:
                                #print(previous_thres, pi, global_iters, paths)
                                df,pthres = read_data(path=path,
                                           test_task='_'.join(dataset.split('_')[:-1]), task=task, pred_thres=pred_thres,
                                           name=name, lr=lr, seed=seed, base=base, full_name=full_name,
                                           return_thres=True, thres=previous_thres[pi], **kwargs)
                                #print('***read data')
                            else:
                                df,pthres = read_data(path=path,
                                           test_task='_'.join(dataset.split('_')[:-1]), task=task, pred_thres=pred_thres,
                                           name=name, lr=lr, seed=seed, base=base, full_name=full_name,
                                           return_thres=True,**kwargs)
                                pred_thres_df.loc[(name, lr, seed, global_iter),'thres'] = pthres
                            all_thres.append(pthres)
                            if dataset not in ds2len:
                                ds2len[dataset] = len(df)
                            elif ds2len[dataset] != len(df):
                                raise ValueError('inconsistent file size {}, {}'.format(base_dir, dataset))

                            dfs.append(df)

                            if dataset == wwo_dataset:
                                if subsets[1] == 'with':
                                    df_w = df.iloc[w_idx]
                                    df_wo = df.iloc[wo_idx]
                                    dfs += [df_w, df_wo]
                                else:
                                    for indice in indices_test:
                                        dfs.append(df[indice])

                            if dataset == wwo_dataset_dev:
                                if subsets[1] == 'with':
                                    df_w = df.iloc[w_idx_dev]
                                    df_wo = df.iloc[wo_idx_dev]
                                    dfs += [df_w, df_wo]
                                else:
                                    for indice in indices_dev:
                                        dfs.append(df[indice])

                            for subset_name, subset_df in zip(subsets[:len(dfs)],dfs):
                                if is_ood:
                                    metrics_func_, metrics_  = metrics_func_ood, metrics_ood
                                elif '_dev' in dataset:
                                    metrics_func_, metrics_ = metrics_func_dev, metrics
                                else:
                                    metrics_func_, metrics_ = metrics_func, metrics
                                met = metrics_func_(subset_df)

                                metrics_ = met if metrics_ is None else metrics_
                                for metric_name in metrics_:
                                    if all_checkpoints:
                                        res_df.loc[(name, lr, seed, global_iter),(dataset, subset_name, metric_name)] = met[metric_name]
                                    else:
                                        res_df.loc[(name, lr, seed),(dataset, subset_name, metric_name)] = met[metric_name]
                            #res_df = res_df.sort_index()

                    except FileNotFoundError:
                        #print(base_dir + 'predictions_{}.csv'.format(dataset), 'not found')
                        pass
                    except ValueError as e:
                        print(e, base_dir, (name, lr, seed, global_iter),(dataset, subset_name, metric_name), [_ for _ in res_df.index])
                        raise
                    previous_thres = all_thres
    if kwargs.get('return_thres_df'):
        return res_df, pred_thres_df
    else:
        return res_df

def aggregate_gab_results(names, lrs, w_idx, wo_idx, pred_thres=None, fprd_agg=False, iptts=False, **kwargs):
    datasets = ['gab_dev', 'gab_test', 'nyt_test']
    metrics = ['f1', 'precision', 'recall', 'fpr','auc']
    if fprd_agg:
        metrics.append('fprd_agg')
    if iptts:
        datasets.pop(-1)
        datasets.append('iptts_test')

    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'gab', 'gab_test', pred_thres, **kwargs)

def aggregate_ws_results(names, lrs, w_idx, wo_idx, pred_thres=None,  fprd_agg=False, iptts=False, **kwargs):
    datasets = ['ws_dev', 'ws_test', 'nyt_test']
    metrics = ['f1', 'precision', 'recall', 'fpr','auc']
    if fprd_agg:
        metrics.append('fprd_agg')
    if iptts:
        datasets.pop(-1)
        datasets.append('iptts_test')
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'ws', 'ws_test', pred_thres, **kwargs)

def aggregate_toxic_results(names, lrs, w_idx, wo_idx, pred_thres=None, **kwargs):
    datasets = ['toxic_dev', 'toxic_test']
    metrics = ['f1', 'precision', 'recall', 'fpr','auc']
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'toxic', 'toxic_test', pred_thres, **kwargs)

def aggregate_biasbios_results(names, lrs, w_idx, wo_idx, pred_thres=None, **kwargs):
    datasets = ['biasbios_dev', 'biasbios_test']
    metrics = ['acc', 'tprd']
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'biasbios', 'biasbios_test', pred_thres, **kwargs)

def aggregate_twitter_sentiment_results(names, lrs, w_idx, wo_idx, wwo_dataset='twitter_sentiment_dev', pred_thres=None, **kwargs):
    datasets = ['twitter_sentiment_dev', 'twitter_sentiment_test']
    metrics = ['acc']
    a, b = ['f1', 'precision', 'recall', 'fpr'], ['positive', 'negative']
    for i in a:
        for j in b:
            metrics.append('{}_{}'.format(i,j))

    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'sentiment', wwo_dataset,
                             metrics_func=get_multi_metric_func(['negative', 'positive']), test_task='twitter_sentiment',
                             **kwargs)

def aggregate_emoji_results(names, lrs, male_ind_test, female_ind_test, aae_ind_test, male_ind_dev, female_ind_dev, aae_ind_dev,
                            pos_label_ids, **kwargs):
    datasets = ['emoji_dev', 'emoji_test']
    subsets = ('full', 'male','female', 'aae')
    metric_func = get_emoji_metric_func(pos_label_ids)
    return aggregate_results(names, lrs, w_idx=None, wo_idx=None, w_idx_dev=None, wo_idx_dev=None, datasets=datasets,
                             subsets=subsets, metrics=['acc','macro_f1', 'fpr', 'fnr'], metrics_func=metric_func, task='emoji',
                             indices_dev=[male_ind_dev, female_ind_dev, aae_ind_dev],
                             indices_test=[male_ind_test, female_ind_test, aae_ind_test],
                             wwo_dataset='emoji_test', wwo_dataset_dev='emoji_test', **kwargs)

def aggregate_twitter_results(names, lrs, w_idx, wo_idx, twitter_task, wwo_dataset='twitter_hate_dev', pred_thres=None, **kwargs):
    datasets = ['twitter_{}_dev'.format(twitter_task), 'twitter_{}_test'.format(twitter_task)] + ['brod2_test']
    metrics = ['f1', 'precision', 'recall', 'fpr','auc']
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, twitter_task, wwo_dataset, pred_thres,
                             metrics_func_ood=metrics_df_ood, metrics_ood=['acc'],
                             **kwargs)

def aggregate_fdcl_results(names, lrs, w_idx, wo_idx, **kwargs):
    datasets = ['fdcl_dev', 'fdcl_test' ,'brod_test']
    metrics = ['acc','macro_f1']
    a, b = ['f1', 'precision', 'recall', 'fpr'], ['overall', 'spam', 'abusive', 'hateful']
    for i in a:
        for j in b:
            metrics.append('{}_{}'.format(i,j))
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'fdcl', 'fdcl_test',
                             metrics_func=get_multi_metric_func(['spam','abusive','hateful']),
                             metrics_func_ood=metrics_df_ood,
                            metrics_ood=['acc'],
                            **kwargs)

def aggregate_dwmw_results(names, lrs, w_idx, wo_idx, **kwargs):
    datasets = ['dwmw_dev', 'dwmw_test' ,'brod3_test']
    metrics = ['acc','fpr','fnr']
    # a, b = ['f1', 'precision', 'recall', 'fpr'], ['overall', 'spam', 'abusive', 'hateful']
    # for i in a:
    #     for j in b:
    #         metrics.append('{}_{}'.format(i,j))
    return aggregate_results(names, lrs, w_idx, wo_idx, datasets, metrics, 'dwmw', 'dwmw_test',
                             metrics_func=dwmw_metric_func,
                             metrics_func_ood=metrics_df_ood,
                             metrics_ood=['acc'],
                             **kwargs)




def aggregate_iptts_results(names, lrs, task, raw_idents, iptts_tokens, pred_thres=None, full_name=False, base='runs-0711/',
                            thres_df=None, all_checkpoints=False, base_task='gab_test', **kwargs):
    seeds = [_ for _ in range(3)]
    if full_name:
        base_dir = '{}/{}_{}/{}/'.format(base, names[0], lrs[0], seeds[0])
    else:
        base_dir = '{}/roberta_{}_{}_{}/{}/'.format(base, task.split('_')[-1], names[0], lrs[0], seeds[0])
    checkpoint_files_ex = get_all_checkpoints(base_dir, base_task)
    checkpoint_global_iters = [_[-1] for _ in checkpoint_files_ex]

    multi_row_index = pd.MultiIndex.from_product([names, lrs, seeds]) if not all_checkpoints \
        else pd.MultiIndex.from_product([names, lrs, seeds, checkpoint_global_iters])

    res_df = pd.DataFrame(index=multi_row_index, columns=['fped','fned'])

    for name in names:
        for lr in lrs:
            for seed in seeds:
                global_iters = thres_df.loc[(name, lr, seed)]
                for global_iter, thres in global_iters.iterrows():

                    if full_name:
                        base_dir = '{}/{}_{}/{}/'.format(base, name, lr, seed)
                    else:
                        base_dir = '{}/roberta_{}_{}_{}/{}/'.format(base, task.split('_')[-1], name, lr, seed)
                    try:
                        df = read_data(path=base_dir + 'predictions_iptts_test.csv', test_task='iptts', task=task,
                                      name=name, lr=lr, seed=seed, base=base, full_name=full_name, thres=thres)
                    except FileNotFoundError:
                        print(base_dir, 'not found')
                        continue
                    #print(len(df), len(iptts_tokens))
                    df['idents'] = iptts_tokens
                    overall_met = metrics_df(df)
                    # group by idents
                    fped, fned = 0, 0
                    for group_name, group_data in df.groupby('idents'):
                        met = metrics_df(group_data)
                        fped += abs(met['fpr'] - overall_met['fpr'])
                        fned += abs(met['recall'] - overall_met['recall'])
                    res_df.loc[(name,lr,seed),'fped'] = fped
                    res_df.loc[(name,lr,seed),'fned'] = fned
    return res_df

def mean_by_seed(results, level=(0,1)):
    return results.groupby(level=level).apply(lambda x: x.mean(axis=0))

def std_by_seed(results, level=(0,1)):
    return results.groupby(level=level).apply(lambda x: x.std(axis=0))

def best_by_dev(results, key=('gab_dev','full','f1')):
    def best_row(group):
        arr = group.loc[:, key].to_numpy()
        #if len(arr[~np.isnan(arr)]) == 0:
        #    best_seed_idx = 0
        #else:
        best_seed_idx = np.argmax(arr)
        return group.iloc[best_seed_idx]
    return results.groupby(level=[0,1]).apply(best_row)

def get_best_lr_and_names(results_merged, method_keys, key=('gab_dev','full','f1')):
    def best_row(group):
        arr = group.loc[:, key].to_numpy()
        if len(arr[~np.isnan(arr)]) == 0:
            best_lr_idx = 0
        else:
            best_lr_idx = np.nanargmax(arr)
        #return group.iloc[best_seed_idx]
        return group.iloc[best_lr_idx].name
    return results_merged.groupby(level=[0]).apply(best_row)

def get_best_ckpt_and_names(results_merged, method_keys, key=('gab_dev','full','f1')):
    def best_row(group):
        arr = group.loc[:, key].to_numpy()
        if len(arr[~np.isnan(arr)]) == 0:
            best_lr_idx = 0
        else:
            best_lr_idx = np.nanargmax(arr)
        #return group.iloc[best_seed_idx]
        return group.iloc[best_lr_idx].name
    return results_merged.groupby(level=[0,1]).apply(best_row)

def get_fix_lr_and_names(results_merged, method_keys, lr_idx, key=('gab_dev','full','f1')):
    def best_row(group):
        #return group.iloc[best_seed_idx]
        return group.iloc[lr_idx].name
    return results_merged.groupby(level=[0]).apply(best_row)

def sanity_check():
    print('asb')

def filter_na(results):
    idx = results.index[results.iloc[:,0].apply(pd.notnull)]
    return results.loc[idx]

def decide_checkpoints_with_early_stopping(df, key, by_lower, iter_per_epoch=None):
    def best_row(group):
        arr = group.loc[:, key]#.to_numpy().astype(np.float32)
        #print(arr)
        #print(np.isnan(arr))
        #print('---')
        if iter_per_epoch:
            steps = []
            for i, (index, row) in enumerate(arr.items()):
                if int(index[-1]) % iter_per_epoch == 0:
                    steps.append(i)
            arr = arr.iloc[steps]
            #print(steps, arr)

        arr_index = arr.index
        arr = arr.to_numpy().astype(np.float32)

        if len(arr[~np.isnan(arr)]) == 0:
            best_lr_idx = 0
        else:
            if by_lower:
                best_lr_idx = np.nanargmin(arr)
            else:
                best_lr_idx = np.nanargmax(arr)
        # return group.iloc[best_seed_idx]
        try:
            return group.loc[arr_index[best_lr_idx]].name
        except Exception:
            print(group.index)
            return group.iloc[0].name

    return df.groupby(level=['method','lr','seed']).apply(best_row)

def extract_epoch_step(best_rows, iter_per_epoch):
    best_steps_epochs = best_rows.apply(lambda row: pd.Series({'step': row[-1], 'epoch': row[-1] // iter_per_epoch - 1}))
    return best_steps_epochs


def load_gab_results(sample_name_or_path, model_names, base='runs-0711/gab', pred_thres=None, idents=None):
    if '/' not in sample_name_or_path:
        df_gab_sa = read_data(base=base, task='gab', name=sample_name_or_path, seed=1, lr='1e-5', split='test',
                              full_name=True)
        df_gab_dev_sa = read_data(base=base, task='gab', name=sample_name_or_path.replace('_test_', '_dev_'), seed=1,
                                  lr='1e-5', split='dev',
                                  full_name=True)
    else:
        df_gab_sa = read_data(base=base, task='gab', path=sample_name_or_path, seed=1, lr='1e-5', split='test',
                              full_name=True)
        df_gab_dev_sa = read_data(base=base, task='gab', path=sample_name_or_path.replace('_test_', '_dev_'),
                                  seed=1, lr='1e-5', split='dev', full_name=True)
    w_gab_sa, wo_gab_sa = partition_by_group_identifiers(idents, df_gab_sa)

    w_gab_dev_sa, wo_gab_dev_sa = partition_by_group_identifiers(idents, df_gab_dev_sa)
    print(len(df_gab_sa), len(df_gab_dev_sa))
    print(len(w_gab_sa) + len(wo_gab_sa), len(w_gab_dev_sa) + len(wo_gab_dev_sa))

    results = aggregate_gab_results(model_names,
                                    ['1e-3', '2e-5', '1e-5', '5e-6'], w_gab_sa, wo_gab_sa, full_name=True, base=base,
                                    max_seed=6, all_checkpoints=True,
                                    w_idx_dev=w_gab_dev_sa, wo_idx_dev=wo_gab_dev_sa, wwo_dataset_dev='gab_dev',
                                    pred_thres=pred_thres)
    results = results.rename_axis(["method", "lr", "seed", 'ts'], axis=0)
    results['gab_test', 'with', 'fprd'] = (
                results['gab_test', 'with', 'fpr'] - results['gab_test', 'full', 'fpr']).abs()
    results['gab_test', 'with', 'fnrd'] = (
                results['gab_test', 'with', 'recall'] - results['gab_test', 'full', 'recall']).abs()
    results['gab_dev', 'with', 'fprd'] = (results['gab_dev', 'with', 'fpr'] - results['gab_dev', 'full', 'fpr']).abs()
    results['gab_dev', 'with', 'fnrd'] = (
                results['gab_dev', 'with', 'recall'] - results['gab_dev', 'full', 'recall']).abs()
    cnt = results.groupby(level=[0, 1]).count()
    results = filter_na(results)
    return results


def get_checkpoints(results, step, save=False, save_postfix='', dev_set_name='gab_dev', base='runs-0711/gab'):
    best_dev_f1_step = decide_checkpoints_with_early_stopping(results, (dev_set_name, 'full', 'f1'), by_lower=False,
                                                              iter_per_epoch=step)
    best_step_epoch = extract_epoch_step(best_dev_f1_step, step)
    best_dev_fairness_step = decide_checkpoints_with_early_stopping(results, (dev_set_name, 'with', 'fprd'),
                                                                    by_lower=True, iter_per_epoch=step)
    best_fairness_step_epoch = extract_epoch_step(best_dev_fairness_step, step)

    if save:
        best_step_epoch.to_csv('{}/gab_best_f1_{}.csv'.format(base, save_postfix))
        best_step_epoch.to_pickle('{}/gab_best_f1_{}.pkl'.format(base, save_postfix))
        best_fairness_step_epoch.to_csv('{}/gab_best_fairness_{}.csv'.format(base, save_postfix))
        best_fairness_step_epoch.to_pickle('{}/gab_best_fairness_{}.pkl'.format(base, save_postfix))
    return best_dev_f1_step, best_step_epoch, best_dev_fairness_step, best_fairness_step_epoch

def get_model_names(base):
    model_names = []
    for item in os.listdir(base):
        if item in ['debug']:
            continue
        if os.path.isdir(os.path.join(base,item)):
            model_name = '_'.join(item.split('_')[:-1])
            if model_name not in model_names:
                model_names.append(model_name)
    return model_names

def compute_fairness(results, task):
    test_set, dev_set = '{}_test'.format(task), '{}_dev'.format(task)
    results = results.rename_axis(["method", "lr", "seed", 'ts'], axis=0)
    results[(test_set,'with','fprd')] = (results[(test_set,'with','fpr')] - results[(test_set, 'full', 'fpr')]).abs()
    results[(test_set,'with','fnrd')] = (results[(test_set,'with','recall')] - results[(test_set, 'full', 'recall')]).abs()
    results[(dev_set,'with','fprd')] = (results[(dev_set,'with','fpr')] - results[(dev_set, 'full', 'fpr')]).abs()
    results[(dev_set,'with','fnrd')] = (results[(dev_set,'with','recall')] - results[(dev_set, 'full', 'recall')]).abs()
    cnt = results.groupby(level=[0,1]).count()
    results = filter_na(results)
    return results