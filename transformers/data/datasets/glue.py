import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
import inspect

import torch
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ...trainer import torch_distributed_zero_first
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures


logger = logging.getLogger(__name__)

@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    processor = None

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate=False,
        do_test=False,
        local_rank=-1,
        processor_args=None,
    ):
        if processor_args is None:
            processor_args = {}

        remove_nw = processor_args.get('remove_nw', False)
        adv_debias = processor_args.get('adv_debias', False)
        bias_attr = processor_args.get('bias_attr', None)

        self.args = args

        processor_args = {k:v for k,v in processor_args.items()}

        processor = glue_processors[args.task_name]()
        if bias_attr:
            processor.set_bias_attr(bias_attr)

        self.processor = processor

        self.output_mode = glue_output_modes[args.task_name]
        # Load data features from cache or dataset file
        real_split = 'train'
        if evaluate:
            if do_test:
                real_split = 'test'
            else:
                real_split = 'dev'
        postfix = []
        if adv_debias:
            postfix.append('_adv')
        if bias_attr:
            postfix.append('_{}'.format(bias_attr))
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                real_split, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name + ''.join(postfix),
            ),
        )
        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not args.overwrite_cache and not remove_nw:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                if evaluate:
                    examples = (
                        processor.get_dev_examples(args.data_dir) if not do_test else processor.get_test_examples(args.data_dir)
                    )
                else:
                    examples = processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                    processor_args=processor_args
                )
                if local_rank in [-1, 0] and not remove_nw:
                    start = time.time()
                    torch.save(self.features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class GLUEMultiTaskDataset(Dataset):
    def __init__(self, task_offset, *datasets):
        self.datasets = datasets
        self.lens = [len(dataset) for dataset in datasets]
        self.MAIN_TASK = 0
        self.task_offset = task_offset

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, i) -> InputFeatures:
        dataset_idx,j = 0,i
        while dataset_idx < len(self.datasets):
            if j < self.lens[dataset_idx]:
                break
            dataset_idx += 1
            j -= self.lens[dataset_idx]
        feat = self.datasets[dataset_idx][j]
        feat.set_task(dataset_idx + self.task_offset)
        return feat


class GLUETaskDatasetWrapper(Dataset):
    def __init__(self, base_dataset, task_id):
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.args = self.base_dataset.args

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, i):
        feat = self.base_dataset[i]
        feat.set_task(self.task_id)
        return feat