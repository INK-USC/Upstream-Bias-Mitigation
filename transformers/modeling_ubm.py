import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy

from transformers.param_utils import TaskLearningArgs, RegularizationArgs, MultiTaskLearningArgs


logger = logging.getLogger(__name__)


class UBMFrameworkAbc:
    def __init__(self, config, classifier_head_cls=None):
        # mtl variables
        self.config = config
        self.mtl_classifiers = nn.ModuleList()

        self.use_head = None

        # adversarial training variables
        self.adv_debias = False
        self.reg_args = None
        self.adv_model = None

        # mtl adversarial learning
        self.mtl_adv_classifiers = nn.ModuleList()
        self.mtl_task_id_to_reg_args = {}

        self.classifier_head_cls = classifier_head_cls

    def add_mtl_classifier(self, num_label):
        tmp_config = copy.deepcopy(self.config)
        tmp_config.num_labels = num_label
        self.mtl_classifiers.append(self.classifier_head_cls(tmp_config))

    def build_mtl(self, mtl_args: MultiTaskLearningArgs):
        # if kwargs.get('label_counts', None):
        #     label_counts = kwargs['label_counts']
        #     for label_count in label_counts:
        #         self.add_mtl_classifier(label_count)
        #     self.use_head = kwargs['use_head']
        if mtl_args.mtl:
            label_counts = mtl_args.label_counts
            for label_count in label_counts:
                self.add_mtl_classifier(label_count)
            self.use_head = mtl_args.use_head
            for targs in mtl_args.mtl_task_args:
                self.build_mtl_adv_training(targs)

    def build_adv_training(self, reg_args: RegularizationArgs):
        assert self.adv_model is None
        self.reg_args = reg_args
        if reg_args.adv_debias:
            self.adv_debias = True
            # self.adv_objective = reg_args.adv_objective
            # self.adv_strength = reg_args.adv_strength
            # self.adv_grad_rev_strength = reg_args.adv_grad_rev_strength
            # self.label_stats = reg_args.label_stats

            from adv_training.model import AdversarialClassifierHead
            self.adv_model = AdversarialClassifierHead(self.config.hidden_size, attr_dim=reg_args.adv_attr_dim,
                                                       adv_objective=reg_args.adv_objective, hidden_layer_num=reg_args.adv_layer_num,
                                                       adv_grad_rev_strength=reg_args.adv_grad_rev_strength)

    def build_mtl_adv_training(self, task_args: TaskLearningArgs):
        task_id = task_args.task_id
        reg_args = task_args.mtl_reg_args
        from adv_training.model import AdversarialClassifierHead

        while len(self.mtl_adv_classifiers) <= task_id - 1: # task id starts from 1
            dummy = nn.Module()
            self.mtl_adv_classifiers.append(dummy)

        self.mtl_adv_classifiers[task_id - 1] = AdversarialClassifierHead(self.config.hidden_size, reg_args.adv_attr_dim,
                                                       adv_objective=reg_args.adv_objective, hidden_layer_num=reg_args.adv_layer_num,
                                                       adv_grad_rev_strength=reg_args.adv_grad_rev_strength)
        self.mtl_task_id_to_reg_args[task_id] = reg_args

