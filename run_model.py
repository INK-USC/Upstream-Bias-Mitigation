# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import json

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from utils.configs import full_configs, combine_args
from transformers.param_utils import MultiTaskLearningArgs, RegularizationArgs
from transformers.trainer import write_verbose_eval_results

logger = logging.getLogger(__name__)


def sanity_check(other_args):
    if other_args.transfer and other_args.mtl:
        raise ValueError('Conflicting option of transfer and mtl')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser.add_argument('--reg_explanations', action='store_true')
    parser.add_argument('--stat_grad', action='store_true')
    parser.add_argument('--stat_grad_method', default='')
    parser.add_argument('--importance_target', default='output', choices=['output','hidden'])
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--early_stop_criterion', type=str, default='loss')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--verbose_test', action='store_true')
    parser.add_argument('--save_less', action='store_true')
    parser.add_argument('--no_eer', action='store_true')
    parser.add_argument('--no_main_objective', action='store_true')

    # transfer learning related parameters
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--load_epoch', default='-1', type=str, help='if -1, load default; if int, load epoch;'
                                                                     'if str, load dir')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--cl_algo', choices=['ewc','lsp'], default=None)
    parser.add_argument('--cl_expl_importance', action='store_true')
    parser.add_argument('--label_num', type=int, nargs='?')
    parser.add_argument('--epoch_len', type=int, default=-1)

    # fresh weights
    parser.add_argument('--fresh_weights', action='store_true')

    # multi task learning
    parser.add_argument('--mtl', action='store_true')
    #parser.add_argument('--mtl_tasks', nargs='*')
    #parser.add_argument('--mtl_data_dirs', nargs='*')
    parser.add_argument('--mtl_args', nargs='*')
    parser.add_argument('--use_head', type=int, help='specify a classifier head to use (for evaluation)')
    parser.add_argument('--mtl_bal_sampling', action='store_true', help='whether to oversample a dataset in an epoch')

    # adversarial debiasing
    parser.add_argument('--adv_debias', action='store_true')
    parser.add_argument('--adv_objective', choices=['eq_odds_laftr', 'eq_odds_ce', 'adv_ce'])
    parser.add_argument('--adv_strength', default=0.1, type=float)
    parser.add_argument('--adv_grad_rev_strength', default=1.0, type=float)
    parser.add_argument('--bias_attr', type=str)
    parser.add_argument('--adv_lr_scale', type=float, default=1.0)
    parser.add_argument('--adv_layer_num', type=int, default=1)

    # add expl reg related args
    # whether run explanation algorithms
    parser.add_argument("--explain", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--neutral_words_file", default='datasets/identity.csv')
    parser.add_argument("--keep_other_nw", default=True)
    parser.add_argument("--softmax_contrib", action='store_true')
    parser.add_argument("--pos_contrib", action='store_true')
    parser.add_argument("--expl_batched", action='store_true')
    parser.add_argument("--expl_batch_size", type=int, default=16)

    # embedding zero out
    parser.add_argument("--do_embedding_zero_out", action='store_true')

    # which algorithm to run
    parser.add_argument("--algo", choices=['soc', 'shap'])

    # the output filename without postfix
    parser.add_argument("--output_filename", default='temp.tmp')

    # see utils/config.py
    parser.add_argument("--use_padding_variant", action='store_true')
    parser.add_argument("--mask_outside_nb", action='store_true')
    parser.add_argument("--nb_range", type=int)
    parser.add_argument("--sample_n", type=int)
    parser.add_argument("--noise_lm", action='store_true')

    # whether use explanation regularization
    parser.add_argument("--reg_strength", type=float)
    parser.add_argument("--reg_abs", action='store_true')

    parser.add_argument("--neg_weight", type=float, default=1.)

    # whether remove neutral words directly at data reading phrase
    parser.add_argument("--remove_nw", action='store_true')

    # if true, generate hierarchical explanations
    parser.add_argument("--hiex", action='store_true')
    parser.add_argument("--hiex_tree_height", default=5, type=int)
    parser.add_argument("--hiex_add_itself", action='store_true')

    parser.add_argument("--lm_dir", default='runs-0711/twitter_sentiment')

    # if configured, only generate explanations for instances with given line numbers
    parser.add_argument("--hiex_idxs", default=None)
    # if true, use absolute values of explanations for hierarchical clustering
    parser.add_argument("--hiex_abs", action='store_true')

    # if either of the two is true, only generate explanations for positive / negative instances
    parser.add_argument("--only_positive", action='store_true')
    parser.add_argument("--only_negative", action='store_true')

    # stop after generating x explanation
    parser.add_argument("--stop", default=100000000, type=int)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        other_args = None
    else:
        model_args, data_args, training_args, other_args = \
            parser.parse_args_into_dataclasses()
        combine_args(full_configs, other_args)
    combine_args(full_configs, model_args)
    combine_args(full_configs, data_args)
    combine_args(full_configs, training_args)

    # sanity check
    sanity_check(other_args)

    if full_configs.debug:
        logger.warning('Debug mode is on')

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        if other_args.label_num is not None:
            num_labels = other_args.label_num
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        output_hidden_states=other_args.stat_grad
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # mtl arguments: each argument will be parsed as a DotDict
    mtl_args = MultiTaskLearningArgs(other_args, training_args, data_args, full_configs, tokenizer)
    main_reg_args = RegularizationArgs(other_args, data_args, full_configs)

    # Get datasets

    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, processor_args={
            'remove_nw': other_args.remove_nw, 'neutral_words_file': other_args.neutral_words_file,
            'tokenizer': tokenizer, 'adv_debias': other_args.adv_debias, 'bias_attr': other_args.bias_attr
        })
        if training_args.do_train
        else None
    )

    # main evaluation set (val or test)
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True,
            do_test=other_args.test, processor_args={
            'remove_nw': other_args.remove_nw, 'neutral_words_file': other_args.neutral_words_file,
            'tokenizer': tokenizer, 'adv_debias': other_args.adv_debias, 'bias_attr': other_args.bias_attr
        })
        if training_args.do_eval
        else None
    )

    # always the test set
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True,
            do_test=True, processor_args={
            'remove_nw': other_args.remove_nw, 'neutral_words_file': other_args.neutral_words_file,
            'tokenizer': tokenizer, 'adv_debias': other_args.adv_debias, 'bias_attr': other_args.bias_attr
        })
        if training_args.do_eval
        else None
    )

    processor = train_dataset.processor if train_dataset is not None else eval_dataset.processor

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        transfer=other_args.transfer,
        freeze=other_args.freeze,
        mtl_args=mtl_args,
        main_reg_args=main_reg_args,
        load_epoch=other_args.load_epoch,
        tokenizer=tokenizer,
        do_embedding_zero_out=other_args.do_embedding_zero_out,
        neutral_words_file=other_args.neutral_words_file
    )

    if other_args.fresh_weights:
        if not training_args.do_train:
            raise ValueError('can not initialize the weights and do eval only')
        logger.info('freshly initializing the weights')
        model.init_weights()

    if other_args.neg_weight != 1:
        logger.info('setting neg class weight to {}'.format(other_args.neg_weight))
        class_weight = torch.FloatTensor([other_args.neg_weight, 1]).cuda()
        model.set_class_weight(class_weight)

    # copy args
    model_args.mtl = data_args.mtl = other_args.mtl

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # # all the mtl arguments
    # mtl_args = {'mtl': other_args.mtl, 'mtl_train_datasets': mtl_train_datasets, 'mtl_eval_datasets': mtl_eval_datasets,
    #             'mtl_test_datasets': mtl_test_datasets, 'mtl_tasks': other_args.mtl_tasks,
    #             'mtl_reg_args': mtl_reg_args, 'mtl_metrics': mtl_metrics}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
        cl_algo=other_args.cl_algo,
        output_dir=training_args.output_dir,
        mtl_args=mtl_args,
        reg_explanations=getattr(full_configs, 'reg_explanations', False),
        tokenizer=tokenizer,
        full_configs=full_configs,
        transfer=other_args.transfer,
        cl_expl_importance=other_args.cl_expl_importance,
        main_reg_args=main_reg_args
    )

    if training_args.do_train:
        wf = open(os.path.join(training_args.output_dir, 'args.json'), 'w')
        json.dump({k:str(v) for k,v in full_configs.__dict__.items()},wf)
        wf.close()

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
            reg_explanations=getattr(full_configs, 'reg_explanations', False),
            configs=full_configs
        )
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        verbose = training_args.do_eval and not training_args.do_train or other_args.verbose_test

        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True,
                            )
            )

        for eval_dataset in eval_datasets:
            split = 'dev' if not other_args.test else 'test'
            if verbose:
                result, raw_results, raw_inputs = trainer.evaluate(eval_dataset=eval_dataset, verbose=True, reg_args=main_reg_args)
                write_verbose_eval_results(training_args, raw_inputs, raw_results, eval_dataset, split, tokenizer,
                                           )
            else:
                result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}_{split}.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
