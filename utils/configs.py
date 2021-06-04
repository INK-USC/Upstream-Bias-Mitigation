"""
This file contain configs for training models and running
"""
import logging, os, json

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        """
        Note: Every arguments here are by default overrode by commandline arguments. (See run_model.py)
        """

        self.raw_data_path = './data/majority_gab_dataset_25k.json'  # default using MAJORITY
        self.data_dir = './data/majority_gab_dataset_25k/'

        # create a classifier head for each item in the list with disjunction of each label
        # for example, [('hd','cv')] means the ground truth label for classification is 1 when hd OR cv is 1.
        # multi-head classifier not implemented. So len(self.label_groups) == 1 should hold.
        self.label_groups = [('hd','cv')]
        self.do_lower_case = True
        self.bert_model = 'bert-base-uncased'

        # for hierarchical explanation algorithms, where to store the language model
        # first time running may require to train a language model
        self.lm_dir = 'runs/lm_model_uncased.pkl'

        # configs for lm in hierarchical explanation algorithms
        self.lm_d_hidden = 1024
        self.lm_d_embed = 300

        # keep batch_size = 1 when using explanation algorithms at SOC
        # You can try to use larger batch_size for GradSHAP by implementing your own code - but
        # GradSHAP requires very intensive GPU memory use and do not runs with a batch_size
        # larger than 1
        self.batch_size = 1

        # context region to be specified for running SOC. Smaller to be faster and larger to be better.
        self.nb_range = 0
        # the number of samples to be drawn for SOC. Smaller to be faster and larger to be better.
        self.sample_n = 1

        # keep self.max_seq_length identical for that training bert. Both are 128 by default
        self.max_seq_length = 128

        # whether pad the words outside the context region of a given phrase to be explained
        # when turned TRUE, SOC yields completely global explanations, and achieve better correlation
        # with word importance captured by linear TF-IDF model.
        # when turned FALSE, SOC explanations are global to its contexts, but local (specific) to the
        # words outside the context region. It has better mathematical interpretation, but achieve lower
        # correlation with word importance captured by linear TF-IDF model
        # NOTE: only configure this with command line
        self.mask_outside_nb = False

        # whether pad the context of the phrase instead of sampling. Turning
        # NOTE: only configure this with command line
        self.use_padding_variant = False

        # neutral words
        self.neutral_words_file = 'data/identity.csv'
        self.reg_mse = True
        self.reg_strength = 0.1

        # whether do training and testing phrase neutral word removal
        self.remove_nw = False

        self.hiex = False

    def update(self, other):
        combine_args(self, other)


full_configs = Config()


def combine_args(args, other):
    # combine and update configs, skip if args.<k> is None
    for k,v in other.__dict__.items():
        if getattr(other, k) is not None:
            if hasattr(args, k) and getattr(args, k) != v:
                logger.info('Overriding {} from {} to {})'.format(k, getattr(args,k), v))
            setattr(args,k,v)
