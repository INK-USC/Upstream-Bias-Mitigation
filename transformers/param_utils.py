import logging
from typing import Dict, Optional, List
import copy
import json

import numpy as np

from .data.datasets import GlueDataset
from .data import (
    glue_compute_metrics,
    glue_tasks_num_labels,
)

from utils.misc import DotDict
from transformers.data.datasets.glue import GLUETaskDatasetWrapper

logger = logging.getLogger(__name__)


class Struct:
    def __getitem__(self, item):
        return self.__dict__[item]


class RegularizationArgs(Struct):
    label_stats = None
    adv_attr_dim = 2
    adv_layer_num = 1

    def __init__(self, some_args, data_args, full_config):
        #self.other_args = other_args
        #self.data_args = data_args

        # root for explanation regularization args
        self.reg_explanations = bool(getattr(some_args, 'reg_explanations', False))
        self.reg_strength = some_args.reg_strength if self.reg_explanations else None
        self.neutral_words_file = some_args.neutral_words_file if self.reg_explanations else None
        self.expl_config = full_config

        # root for adv debiasing args
        self.adv_debias = bool(getattr(some_args, 'adv_debias', False))
        self.adv_objective = some_args.adv_objective if self.adv_debias else None
        self.adv_strength = some_args.adv_strength if self.adv_debias else None
        self.adv_grad_rev_strength = some_args.adv_grad_rev_strength if self.adv_debias else None

        if hasattr(some_args, 'adv_layer_num'):
            self.adv_layer_num = some_args.adv_layer_num

            # self.label_stats =  processor.get_label_stats(self.data_args.data_dir, 'train') if other_args.adv_debias else None,


class TaskLearningArgs(Struct):
    """

    """
    mtl_train_dataset = None
    mtl_eval_dataset = None
    mtl_test_dataset = None
    mtl_metric = None
    mtl_task_name = None
    mtl_reg_args: Optional[RegularizationArgs] = None
    mtl_metric_func = None
    mtl_label_count: int
    mtl_data_dir: str

    task_id: int

    def __init__(self, task_id, base_data_args, training_args, other_args, full_config, tokenizer):
        self.task_id = task_id
        self.data_args = copy.deepcopy(base_data_args)
        self.training_args = training_args
        self.other_args = other_args
        self.full_config = full_config
        self.tokenizer = tokenizer

    def load_from_json_str(self, task_args_str):
        dic = json.loads(task_args_str.replace("'",'"'))
        # required_keys = ['mtl_data_dir', 'mtl_task_name']
        # for k in required_keys:
        #    assert k in dic
        # for k, v in dic.items():
        #    setattr(self, k, v)
        self.mtl_data_dir = dic['mtl_data_dir']
        self.mtl_task_name = dic['mtl_task_name']

        if 'mtl_reg_args' in dic:
            reg_args_obj = DotDict(**dic['mtl_reg_args'])
            self.mtl_reg_args = RegularizationArgs(reg_args_obj, self.data_args, self.full_config)

        self.data_args.task_name = self.mtl_task_name
        self.data_args.data_dir = self.mtl_data_dir
        self.mtl_metric_func = self.get_compute_metrics_func(self.mtl_task_name)

        self.mtl_label_count = glue_tasks_num_labels[self.mtl_task_name]

        self._construct_datasets_for_task()

    def get_compute_metrics_func(self, task_name):
        def compute_metrics(p) -> Dict:
            preds = np.argmax(p.predictions, axis=1)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics

    def __getitem__(self, item):
        return self.__dict__[item]

    def _construct_datasets_for_task(self):
        reg_args = self.mtl_reg_args
        tokenizer = self.tokenizer
        data_args, training_args, other_args = self.data_args, self.training_args, self.other_args
        train_mtl_ds = (
            GLUETaskDatasetWrapper(
                GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, processor_args={
                    'remove_nw': other_args.remove_nw,
                    'neutral_words_file': getattr(reg_args, 'neutral_words_file', ''),
                    'tokenizer': tokenizer, 'adv_debias': getattr(reg_args, 'adv_debias', False),
                    'bias_attr': getattr(reg_args, 'bias_attr', None)
                }), task_id=self.task_id)
            if training_args.do_train
            else None
        )
        eval_mtl_ds = (
            GLUETaskDatasetWrapper(
                GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True,
                            do_test=other_args.test, processor_args={
                        'remove_nw': other_args.remove_nw,
                        'neutral_words_file': getattr(reg_args, 'neutral_words_file', ''),
                        'tokenizer': tokenizer, 'adv_debias': getattr(reg_args, 'adv_debias', False),
                        'bias_attr': getattr(reg_args, 'bias_attr', None)
                    }), task_id=self.task_id)
            if training_args.do_eval
            else None
        )
        test_mtl_ds = (
            GLUETaskDatasetWrapper(
                GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True,
                            do_test=True, processor_args={
                        'remove_nw': other_args.remove_nw,
                        'neutral_words_file': getattr(reg_args, 'neutral_words_file', ''),
                        'tokenizer': tokenizer, 'adv_debias': getattr(reg_args, 'adv_debias', False),
                        'bias_attr': getattr(reg_args, 'bias_attr', None)
                    }), task_id=self.task_id)
            if training_args.do_eval
            else None
        )
        self.mtl_train_dataset = train_mtl_ds
        self.mtl_eval_dataset = eval_mtl_ds
        self.mtl_test_dataset = test_mtl_ds


class MultiTaskLearningArgs:
    def __init__(self, other_args, training_args, base_data_args, full_config, tokenizer):
        """
        Requires --mtl_args in the command line, a list of items
        {'mtl_data_dir': xxx, 'mtl_task_name': xxx, 'mtl_reg_args': {'reg_explanations': 0, 'adv_debias': 1, \
        'adv_objective': xxx, 'adv_strength': xxx, 'adv_grad_rev_strength': xxx}}

        Single quotes will be automatically converted to double quotes
        """

        self.mtl = other_args.mtl
        self.tokenizer = tokenizer
        self.mtl_task_args: List[TaskLearningArgs] = []
        self.task2id = {}
        self.task_id_offset = 1  # TASK ID STARTS FROM 1

        if other_args.mtl_args:
            for i, mtl_reg_args_str in enumerate(other_args.mtl_args):
                task_args = TaskLearningArgs(task_id=i + self.task_id_offset, base_data_args=base_data_args, full_config=full_config,
                                             training_args=training_args, other_args=other_args,
                                             tokenizer=tokenizer)
                task_args.load_from_json_str(mtl_reg_args_str)
                self.mtl_task_args.append(task_args)
                self.task2id[task_args.mtl_task_name] = i + self.task_id_offset

        self.label_counts = [targs.mtl_label_count for targs in self.mtl_task_args]
        self.use_head = other_args.use_head

        self.mtl_tasks = [targs.mtl_task_name for targs in self.mtl_task_args]
        self.mtl_metrics = [targs.mtl_metric_func for targs in self.mtl_task_args]
        self.mtl_reg_args = [targs.mtl_reg_args for targs in self.mtl_task_args]

    def get_datasets(self, set_name):
        if set_name == 'train':
            ret = [targs.mtl_train_dataset for targs in self.mtl_task_args]
        elif set_name == 'eval':
            ret = [targs.mtl_eval_dataset for targs in self.mtl_task_args]
        elif set_name == 'test':
            ret = [targs.mtl_eval_dataset for targs in self.mtl_task_args]
        else:
            raise ValueError
        return ret

    def get_targs_by_id(self, task_id):
        return self.mtl_task_args[task_id - self.task_id_offset]

    def do_expl_regs_any(self):
        for targs in self.mtl_task_args:
            rargs = targs.mtl_reg_args
            if rargs.reg_explanations:
                return True
        return False

    def __getitem__(self, item):
        return getattr(self, item)