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
""" GLUE processors and helpers """

import logging
import os
from enum import Enum
from typing import List, Optional, Union

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures, json
import csv

import numpy as np
import pandas as pd

try:
    from utils.misc import find_neutral_word_positions, get_possible_nw_tokens
except Exception as e:
    print(e)

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
        processor_args=None
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task,
                                                     processor_args=processor_args)
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode,
        processor_args=processor_args
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
            examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


def _glue_convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
        processor_args=None
):
    if max_length is None:
        max_length = tokenizer.max_len

    if processor_args.get('remove_nw', False):
        remove_nw = True
        neutral_words, _ = get_possible_nw_tokens(processor_args['neutral_words_file'], tokenizer)
        def filter_func(tokens):
            spans = find_neutral_word_positions(tokens, neutral_words, tokenizer)
            # regions to be masked
            masked_pos = set()
            for span in spans:
                for i in range(span[0], span[1] + 1):
                    masked_pos.add(i)
            filtered_tokens = []
            for i, token in enumerate(tokens):
                if i not in masked_pos:
                    filtered_tokens.append(token)
            return filtered_tokens
    else:
        filter_func = None

    if task is not None:
        processor = glue_processors[task]()

        bias_attr = processor_args.get('bias_attr', None)
        if bias_attr:
            processor.set_bias_attr(bias_attr)

        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
        filter_func=filter_func
    )


    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i], attr=examples[i].attr)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def twitter_processor(twitter_task):
    label_space = {
        'hate': [0, 1],
        'sentiment': [0, 1, 2],
        'offensive': [0,1],
        'emoji': [_ for _ in range(20)]
    }

    class TwittertProcessor(DataProcessor):
        """Processor for the twitter sentiment classification data set (GLUE version)."""

        def get_example_from_tensor_dict(self, tensor_dict):
            """See base class."""
            return InputExample(
                tensor_dict["idx"].numpy(),
                tensor_dict["sentence"].numpy().decode("utf-8"),
                None,
                str(tensor_dict["label"].numpy()),
            )

        def get_label_stats(self, data_dir, split):
            arr = np.zeros((2, 2))
            df = pd.read_csv(os.path.join(data_dir, 'aae_%s.tsv' % split), sep='\t')
            df = df.iloc[1:]  # mistake, but kept
            arr[0, 0] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] <= 0])
            arr[0, 1] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] > 0])
            arr[1, 0] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] <= 0])
            arr[1, 1] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] > 0])
            return arr

        def get_examples(self, data_dir, split, ident):
            """See base class."""
            if os.path.isfile(os.path.join(data_dir,'aae_{}.tsv'.format(split))):
                f = open(os.path.join(data_dir, 'aae_%s.tsv' % split))
                # else:
                #    f = open(os.path.join(data_dir, '%s_subset.csv' % split))
                reader = csv.DictReader(f, delimiter='\t')
                next(reader)  # skip header # mistake, but kept
                examples = []
                for i, row in enumerate(reader):
                    example = InputExample(text_a=row['text'], guid='%s-%s' % (split, i))
                    label = int(row['label'])
                    example.label = label
                    example.attr = 1 if float(row['aae_prob']) > 0 else 0
                    examples.append(example)
                f.close()
                return examples
            else:
                with open(os.path.join(data_dir, '{}_text.txt'.format(split))) as f:
                    text = [_.strip() for _ in f.readlines()]
                with open(os.path.join(data_dir, '{}_labels.txt').format(split)) as f:
                    labels = [int(_.strip()) for _ in f.readlines()]
                return self._create_examples(text, labels, ident)

        def get_train_examples(self, data_dir):
            return self.get_examples(data_dir, 'train', 'train')

        def get_dev_examples(self, data_dir):
            return self.get_examples(data_dir, 'val', 'dev')

        def get_test_examples(self, data_dir):
            return self.get_examples(data_dir, 'test', 'test')

        def get_labels(self):
            """See base class."""
            return label_space[twitter_task]

        def _create_examples(self, texts, labels, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for i, (text, label) in enumerate(zip(texts, labels)):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = text
                label = label
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples

    return TwittertProcessor

class WSProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        # f = open(os.path.join(data_dir, '%s.tsv' % split))
        # reader = csv.reader(f, delimiter='\t')
        # next(reader) # skip header
        # examples = []
        # for i, row in enumerate(reader):
        #     example = InputExample(text_a=row[1], guid='%s-%s' % (split, i))
        #     label = int(row[2])
        #     example.label = label
        #     examples.append(example)
        # f.close()
        examples = []
        df = pd.read_csv(os.path.join(data_dir, '%s_adv.csv' % split)).dropna()
        for i, row in df.iterrows():
            if row['text']:
                example = InputExample(text_a=row['text'], label=row['is_hate'], guid='%s-%s' % (split, i))
                example.attr = int(row['a'])
                examples.append(example)


        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]


class WikiToxicProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        f = open(os.path.join(data_dir, '%s.tsv' % split))
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        examples = []
        for i, row in enumerate(reader):
            example = InputExample(text_a=row[1], guid='%s-%s' % (split, i))
            label = int(row[-1])
            example.label = label
            examples.append(example)
        f.close()
        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RemoveNeutralWordsProcessor:
    def __init__(self, remove_nw=False, neutral_word_file=None, tokenizer=None):
        self.neutral_word_file=neutral_word_file
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw
        if remove_nw:
            self.neutral_words = get_possible_nw_tokens(self.neutral_word_file, tokenizer, tokenize=False)


class GabProcessor(DataProcessor, RemoveNeutralWordsProcessor):
    def __init__(self, remove_nw=False, neutral_word_file=None, tokenizer=None, subset=None):
        super().__init__(remove_nw, neutral_word_file, tokenizer)
        self.label_groups = [['hd','cv']]

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """

        examples = []
        # try:
        #     f = open(os.path.join(data_dir, '%s.jsonl' % split))
        #     for i, line in enumerate(f.readlines()):
        #         data = json.loads(line)
        #         example = InputExample(text_a=data['Text'], guid='%s-%s' % (split, i))
        #
        #         for j, label_group in enumerate(self.label_groups):
        #             tn = 0
        #             for key in label_group:
        #                 tn += int(data[key])
        #             setattr(example, 'label' if j == 0 else 'label_%d' % j, 1 if tn else 0)
        #         if label is None or example.label == label:
        #             examples.append(example)
        # except FileNotFoundError:
        df = pd.read_csv(os.path.join(data_dir, '%s_adv.csv' % split)).dropna()
        for i, row in df.iterrows():
            if row['text']:
                example = InputExample(text_a=row['text'], label=row['label'], guid='%s-%s' % (split, i))
                example.attr = int(row['a'])
                examples.append(example)

        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]


class NYTProcessor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        f = open(os.path.join(data_dir, '%s.csv' % split))
        #else:
        #    f = open(os.path.join(data_dir, '%s_subset.csv' % split))
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        examples = []
        for i, row in enumerate(reader):
            example = InputExample(text_a=row[1], guid='%s-%s' % (split, i))
            label = int(row[2])
            example.label = label
            examples.append(example)
        f.close()
        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]


class FDCL18Processor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None, subset=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        f = open(os.path.join(data_dir, 'aae_%s.tsv' % split))

        reader = csv.DictReader(f, delimiter='\t')
        next(reader) # skip header # mistake, but kept
        examples = []
        for i, row in enumerate(reader):
            example = InputExample(text_a=row['text'], guid='%s-%s' % (split, i))
            label = int(row['label'])
            example.label = label
            example.attr = 1 if float(row['aae_prob']) > 0 else 0
            examples.append(example)
        f.close()
        return examples

    def get_label_stats(self, data_dir, split):
        arr = np.zeros((2,2))
        df = pd.read_csv(os.path.join(data_dir, 'aae_%s.tsv' % split), sep='\t')
        df = df.iloc[1:] # mistake, but kept
        arr[0, 0] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] <= 0])
        arr[0, 1] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] > 0])
        arr[1, 0] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] <= 0])
        arr[1, 1] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] > 0])
        return arr

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1,2,3]


class DWMWProcessor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None, subset=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw
        print('**Make sure label 0 is non-hate**')

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        df = pd.read_csv(data_dir + '{}.csv'.format(split))
        examples = []
        for i, row in df.iterrows():
            example = InputExample(text_a=row['text'], guid='%s-%s' % (split, i))
            label = row['label']
            example.label = label
            example.attr = 1 if row['aae_prob'] > 0 else 0
            examples.append(example)
        return examples

    def get_label_stats(self, data_dir, split):
        arr = np.zeros((2,2))
        df = pd.read_csv(os.path.join(data_dir, 'aae_%s.tsv' % split), sep='\t')
        df = df.iloc[1:] # mistake, but kept
        arr[0, 0] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] <= 0])
        arr[0, 1] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] > 0])
        arr[1, 0] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] <= 0])
        arr[1, 1] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] > 0])
        return arr

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1,2,3]


class BiasBiosProcessor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None, subset=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        df = pd.read_csv(os.path.join(data_dir, '{}.csv'.format(split)))
        examples = []
        for i, row in df.iterrows():
            example = InputExample(text_a=row['hard_text'], guid='%s-%s' % (split, i))
            label = row['p_id']
            example.label = label
            example.attr = 0 if row['g'] == 'm' else 1
            if (row['g'] not in ['m','f']):
                print(row['g'])
            examples.append(example)
        return examples

    def get_label_stats(self, data_dir, split):
        arr = np.zeros((2,2))
        # dummy
        return arr

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [_ for _ in range(28)]



class IPTTSProcessor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        f = open(os.path.join(data_dir, '%s.csv' % split))
        #else:
        #    f = open(os.path.join(data_dir, '%s_subset.csv' % split))
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        examples = []
        for i, row in enumerate(reader):
            example = InputExample(text_a=row[1], guid='%s-%s' % (split, i))
            label = int(row[-1])
            example.label = label
            examples.append(example)
        f.close()
        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]


class BROD16Processor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None, subset=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw
        print('**Make sure label 0 is non-hate**')

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        df = pd.read_csv('datasets/BROD16/test.csv')
        examples = []
        for i, row in df.iterrows():
            example = InputExample(text_a=row['text'], guid='%s-%s' % (split, i))
            label = 0 # always non hate
            example.label = label
            example.attr = 1 # always aae
            examples.append(example)
        return examples

    def get_label_stats(self, data_dir, split):
        arr = np.zeros((2,2))
        df = pd.read_csv(os.path.join(data_dir, 'aae_%s.tsv' % split), sep='\t')
        df = df.iloc[1:] # mistake, but kept
        arr[0, 0] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] <= 0])
        arr[0, 1] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] > 0])
        arr[1, 0] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] <= 0])
        arr[1, 1] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] > 0])
        return arr

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1,2,3]

class EmojiProcessor(DataProcessor):
    def __init__(self, remove_nw=False, neutral_words=None, tokenizer=None):
        self.neutral_words = neutral_words
        self.tokenizer = tokenizer
        self.remove_nw = remove_nw
        self.label2id, self.id2label = None, None
        self.bias_attr = None

    def set_bias_attr(self, bias_attr):
        self.bias_attr = bias_attr

    def _load_label_mapping(self, data_dir):
        with open(os.path.join(data_dir, 'mapping.csv')) as f:
            rows = f.readlines()
        labels = [_.strip() for _ in rows]
        assert len(labels) == 10
        self.id2label = labels
        self.label2id = {label: i for (i,label) in enumerate(labels)}

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        #if not self.subset:
        if not self.label2id:
            self._load_label_mapping(data_dir)
        df = pd.read_csv(os.path.join(data_dir, '%s.csv' % split), quoting=csv.QUOTE_ALL)
        examples = []
        for i, row in df.iterrows():
            text = row['text']
            label = self.label2id[row['label']]
            bin_gender = row['gender']
            is_aae = 1 if row['aae_prob'] > 0 else 0

            example = InputExample(text_a=text, guid='%s-%s' % (split, i))
            example.label = label
            if self.bias_attr == 'aae':
                example.attr = is_aae
            elif self.bias_attr == 'gender':
                example.attr = bin_gender
            examples.append(example)
        return examples

    def get_label_stats(self, data_dir, split):
        arr = np.zeros((2,2))
        df = pd.read_csv(os.path.join(data_dir, 'aae_%s.tsv' % split), sep='\t')
        df = df.iloc[1:] # mistake, but kept
        arr[0, 0] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] <= 0])
        arr[0, 1] = len(df.loc[df['label'] == 0].loc[df['aae_prob'] > 0])
        arr[1, 0] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] <= 0])
        arr[1, 1] = len(df.loc[df['label'] != 0].loc[df['aae_prob'] > 0])
        return arr

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1,2,3,4,5,6,7,8,9]



glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,

    "twitter_sentiment": 3,
    "twitter_hate": 2,
    "twitter_offensive": 2,
    'twitter_emoji': 20,

    'gab': 2,
    'nyt': 2,
    'ws': 2,

    'fdcl': 4,
    'toxic': 2,
    'iptts': 2,
    'emoji': 10,
    'brod': 4,
    "dwmw": 3,
    'biasbios': 28
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,

    "twitter_sentiment": twitter_processor('sentiment'),
    'twitter_hate': twitter_processor('hate'),
    'twitter_offensive': twitter_processor('offensive'),
    'twitter_emoji': twitter_processor('emoji'),

    'gab': GabProcessor,
    'nyt': NYTProcessor,

    'fdcl': FDCL18Processor,
    "ws": WSProcessor,

    "toxic": WikiToxicProcessor,
    "iptts": IPTTSProcessor,
    "emoji": EmojiProcessor,
    "brod": BROD16Processor,
    "dwmw": DWMWProcessor,
    'biasbios': BiasBiosProcessor
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",

    "twitter_sentiment": "classification",
    "twitter_hate": "classification",
    'twitter_offensive': "classification",
    'twitter_emoji': "classification",

    "gab": "classification",
    "nyt": "classification",
    "fdcl": "classification",
    "ws": "classification",

    "toxic": "classification",
    "iptts": "classification",
    "emoji": "classification",
    "brod": "classification",
    "dwmw": "classification",
    'biasbios': "classification"
}

