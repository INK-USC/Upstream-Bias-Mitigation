import json
import logging
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, DefaultDataCollator
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from .training_args import TrainingArguments, is_tpu_available

from hiex import SamplingAndOcclusionExplain
import random, csv
#from .param_utils import RegularizationArgs
from utils.metric_utils import compute_metrics_from_csv

from sklearn.metrics import f1_score

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


def get_crit_metrics(task):
    if task in ['fdcl']:
        crit_metrics = ['f1', 'macro_f1', 'fprd']
    elif task in ['emoji','biasbios']:
        crit_metrics = ['f1']
    else:
        crit_metrics = ['f1','fprd']
    return crit_metrics

def higher_better(metric_name):
    return metric_name not in ['fprd','loss']

def update_crit_metrics(record_dict, metric_dic, task):
    if not record_dict:
        #record_dict = {k: None for k in get_crit_metrics(task)}
        for k in get_crit_metrics(task):
            if higher_better(k):
                record_dict[k] = -1e10
            else:
                record_dict[k] = 1e10
    best_mets = []
    for k in record_dict:
        if metric_dic[k] > record_dict[k] and higher_better(k):
            best_mets.append(k)
            record_dict[k] = metric_dic[k]
        elif metric_dic[k] < record_dict[k] and not higher_better(k):
            best_mets.append(k)
            record_dict[k] = metric_dic[k]
    return best_mets


def update_mtl_crit_metrics(record_dict_overall, record_dicts):
    if not record_dict_overall:
        for record_dict in record_dicts:
            for k in record_dict:
                record_dict_overall[k] = -1e10 if higher_better(k) else 1e10
    best_mets = []
    for k in record_dict_overall:
        all_values = [record_dict[k] for record_dict in record_dicts if k in record_dict]
        mean_value = np.mean(all_values)
        if (mean_value > record_dict_overall[k] and higher_better(k)) \
        or (mean_value < record_dict_overall[k] and not higher_better(k)):
            best_mets.append(k)
            record_dict_overall[k] = mean_value
    return  best_mets

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def write_verbose_eval_results(training_args, raw_inputs, raw_results, eval_dataset, split, tokenizer,
                               epoch=None, global_iter=None):
    post_fix = ''
    if epoch is not None:
        post_fix += '_epoch_{}'.format(epoch)
    if global_iter is not None:
        post_fix += '_global_{}'.format(global_iter)
    filename = os.path.join(training_args.output_dir,
                     'predictions_{}_{}{}.csv'.format(eval_dataset.args.task_name, split, post_fix))
    wf = open(filename, 'w')
    rows = []
    for i, input_feature in enumerate(raw_inputs.features):
        input_tokens = tokenizer.convert_ids_to_tokens(input_feature.input_ids)
        input_tokens = [_ for _ in input_tokens if _ not in ['<pad>']]
        label = input_feature.label
        pred = raw_results.predictions[i]
        rows.append([' '.join(input_tokens), label] + pred.tolist())
    writer = csv.writer(wf)
    writer.writerows(rows)
    wf.close()
    return filename


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for the first one (locally) to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class MTLDataloader(DataLoader):
    def __init__(self, dummy_dataset, *args, **kwargs):
        data_loaders = kwargs.pop('data_loaders')
        mtl_bal_sampling = kwargs.pop('mtl_bal_sampling')
        super().__init__(dummy_dataset, *args, **kwargs)
        self.data_loaders = data_loaders
        self.mtl_bal_sampling = mtl_bal_sampling

    def __iter__(self):
        def mtl_data_iterator():
            draws = []
            for i in range(len(self.data_loaders)):
                draws.extend([i] * len(self.data_loaders[i]))
            iterators = [iter(_) for _ in self.data_loaders]
            random.Random(0).shuffle(draws)
            for loader_id in draws:
                iterator = iterators[loader_id]
                yield next(iterator)

        def mtl_bal_data_iterator():
            draws = []
            max_dataloader_len = max([len(x) for x in self.data_loaders])
            for i in range(len(self.data_loaders)):
                draws.extend([i] * max_dataloader_len)
            iterators = [iter(_) for _ in self.data_loaders]
            random.Random(0).shuffle(draws)
            for loader_id in draws:
                iterator = iterators[loader_id]
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[loader_id] = iter(self.data_loaders[loader_id])
                    iterator = iterators[loader_id]
                    batch = next(iterator)
                yield batch

        if self.mtl_bal_sampling:
            return mtl_bal_data_iterator()
        else:
            return mtl_data_iterator()

    def __len__(self):
        if self.mtl_bal_sampling:
            max_dataloader_len = max([len(x) for x in self.data_loaders])
            return max_dataloader_len * len(self.data_loaders)
        else:
            return sum([len(_) for _ in self.data_loaders])


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """
    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None

    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            transfer,
            main_reg_args,
            reg_explanations,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            prediction_loss_only=False,
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            cl_algo=None,
            output_dir=None,
            mtl_args=None,
            full_configs=None,
            tokenizer=None,
            cl_expl_importance=True,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        from .param_utils import TaskLearningArgs, MultiTaskLearningArgs, RegularizationArgs
        self.model = model
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if not is_wandb_available():
            logger.info(
                "You are instantiating a Trainer but wandb is not installed. Install it to use Weights & Biases logging."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_local_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

        self.cl_algo = cl_algo
        self.mtl_args: MultiTaskLearningArgs = mtl_args
        self.transfer = transfer

        self.save_all = False
        # if transfer learning, do not save all checkpoints
        if self.transfer:
            self.save_all = False

        self.save_eer = not full_configs.no_eer
        logger.info('save eer is {}'.format(self.save_eer))

        self.mtl = self.mtl_args.mtl
        self.main_reg_args: RegularizationArgs = main_reg_args
        self.tokenizer = tokenizer

        self.no_main_objective = full_configs.no_main_objective

        self.full_configs = full_configs
        self.epoch_len = self.full_configs.epoch_len

        if self.mtl:
            # from .data.datasets.glue import GLUEMultiTaskDataset
            self.all_train_datasets = [train_dataset] + self.mtl_args.get_datasets('train')
            self.all_eval_datasets = [eval_dataset] + self.mtl_args.get_datasets('eval')
            self.all_test_datasets = [test_dataset] + self.mtl_args.get_datasets('test')

            # self.all_train_datasets = [GLUEMultiTaskDataset(i, train_ds) for i,train_ds in enumerate(all_train_ds) if train_ds is not None]
            # self.all_sep_eval_datasets = all_eval_ds #[GLUEMultiTaskDataset(i, eval_ds) for i, eval_ds in enumerate(all_eval_ds) if eval_ds is not None]
            # self.all_sep_test_datasets = all_test_ds #[GLUEMultiTaskDataset(i, eval_ds) for i, eval_ds in enumerate(all_eval_ds) if eval_ds is not None]

        self.stat_grad = self.full_configs.stat_grad
        self.stat_grad_method = self.full_configs.stat_grad_method
        self.wf_stat_grad = None

        if reg_explanations: # or self.stat_grad:
            train_lm_dataloader, dev_lm_dataloader = self.get_eval_dataloader(), self.get_eval_dataloader() # hack
            self.explainer = SamplingAndOcclusionExplain(model, full_configs, tokenizer, device='cuda',
                                                         vocab=tokenizer.encoder,
                                                         train_dataloader=train_lm_dataloader,
                                                         dev_dataloader=dev_lm_dataloader,
                                                         lm_dir=full_configs.lm_dir,
                                                         output_path=os.path.join(full_configs.output_dir,
                                                                                  'explanation.txt'),

                                                         )
        else:
            self.explainer = None


        if self.stat_grad:
            filename = os.path.join(self.full_configs.output_dir,
                                    'reg_info_{}.txt'.format(self.full_configs.importance_target))
            self.wf_stat_grad = open(filename,'w')

        # add explainer for tasks
        self.mtl_explainers = {}
        for i, targs in enumerate(self.mtl_args.mtl_task_args):
            task_id = i + self.mtl_args.task_id_offset
            if targs.mtl_reg_args.reg_explanations:
                full_configs_task = copy.deepcopy(full_configs)
                full_configs_task.reg_strength = targs.mtl_reg_args.reg_strength
                full_configs_task.neutral_words_file = targs.mtl_reg_args.neutral_words_file

                self.mtl_explainers[task_id] = SamplingAndOcclusionExplain(model, full_configs_task, tokenizer, device='cuda',
                                                         vocab=tokenizer.encoder,
                                                         train_dataloader=None,
                                                         dev_dataloader=None,
                                                         lm_dir=full_configs.lm_dir,
                                                         output_path=os.path.join(full_configs.output_dir,
                                                                                  'explanation.txt'),
                                                         )

        if self.cl_algo is not None:
            from cl_methods.wrapper import CLWrapper
            if self.cl_algo in ['ewc', 'lsp']:
                self.cl_train_data_loader = self.get_train_dataloader()
                self.cl_wrapper = CLWrapper(self.model, method=self.cl_algo, output_dir=output_dir,
                                            explainer=self.explainer, expl_importance=cl_expl_importance,
                                            src_dir=full_configs.model_name_or_path)
            else:
                raise ValueError

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        if self.mtl:
            _tmp_data_loaders = [
                DataLoader(
                    ds,
                    batch_size=self.args.train_batch_size,
                    sampler=(
                        RandomSampler(ds)
                        if self.args.local_rank == -1
                        else DistributedSampler(ds)
                    ),
                    collate_fn=self.data_collator.collate_batch,
                ) for ds in self.all_train_datasets
            ]
            data_loader = MTLDataloader(
                self.train_dataset,  # dummy
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
                data_loaders=_tmp_data_loaders,
                mtl_bal_sampling=self.full_configs.mtl_bal_sampling
            )
        else:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        sampler = get_tpu_sampler(eval_dataset) if is_tpu_available() else None

        # if self.mtl:
        #     if eval_dataset is None:
        #         all_eval_datasets = self.all_eval_datasets
        #     else:
        #         all_eval_datasets = [eval_dataset]
        #     _tmp_data_loaders = [
        #             DataLoader(
        #                 ds,
        #                 batch_size=self.args.eval_batch_size,
        #                 sampler=get_tpu_sampler(ds) if is_tpu_available() else None,
        #                 collate_fn=self.data_collator.collate_batch,
        #                 shuffle=False
        #             ) for ds in all_eval_datasets
        #         ]
        #
        #     data_loader = MTLDataloader(
        #         eval_dataset if eval_dataset is not None else self.eval_dataset,
        #         sampler=sampler,
        #         batch_size=self.args.eval_batch_size,
        #         shuffle=False,
        #         collate_fn=self.data_collator.collate_batch,
        #         data_loaders=_tmp_data_loaders)
        # else:
        data_loader = DataLoader(
            eval_dataset if eval_dataset is not None else self.eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        sampler = get_tpu_sampler(test_dataset) if is_tpu_available() else None

        # if self.mtl:
        #     if eval_dataset is None:
        #         all_eval_datasets = self.all_eval_datasets
        #     else:
        #         all_eval_datasets = [eval_dataset]
        #     _tmp_data_loaders = [
        #             DataLoader(
        #                 ds,
        #                 batch_size=self.args.eval_batch_size,
        #                 sampler=get_tpu_sampler(ds) if is_tpu_available() else None,
        #                 collate_fn=self.data_collator.collate_batch,
        #                 shuffle=False
        #             ) for ds in all_eval_datasets
        #         ]
        #
        #     data_loader = MTLDataloader(
        #         eval_dataset if eval_dataset is not None else self.eval_dataset,
        #         sampler=sampler,
        #         batch_size=self.args.eval_batch_size,
        #         shuffle=False,
        #         collate_fn=self.data_collator.collate_batch,
        #         data_loaders=_tmp_data_loaders)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_optimizers(
            self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.


        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        specials = ['adv_']

        # set separate lr for adv learning modules
        regular_named_parameters = [(n,p) for n,p in self.model.named_parameters() if not any(x in n for x in specials)]
        adv_parameters =[(n,p) for n,p in self.model.named_parameters() if any(x in n for x in specials)]

        if adv_parameters:
            logger.info('Parameters for adversarial learning {}'.format([x[0] for x in adv_parameters]))

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in regular_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in regular_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if adv_parameters:
            optimizer_grouped_parameters.append({
                'params': [p for n, p in adv_parameters],
                'initial_lr': self.args.learning_rate * self.full_configs.adv_lr_scale
            })

        if not self.cl_algo:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        elif self.cl_algo in ['ewc','lsp']:
            from cl_methods.regularizers import Weight_Regularized_AdamW
            optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,
                                                 eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.
        """
        wandb.init(name=self.args.logging_dir, config=vars(self.args))
        # keep track of model topology and gradients
        wandb.watch(self.model)

    def train(self, reg_explanations, model_path: Optional[str] = None, configs=None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            if not configs.transfer:
                optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        model.to(self.args.device)
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})
        if is_wandb_available():
            self._setup_wandb()

        # Train!
        if is_tpu_available():
            num_examples = len(train_dataloader._loader._loader.dataset)
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            num_examples = len(train_dataloader.dataset)
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        patience = configs.early_stop
        criterion = configs.early_stop_criterion

        best_result = -1e10
        exit_flag = False

        main_record_dict = {}
        mtl_record_dicts = {task: {} for task in self.mtl_args.mtl_tasks}
        mtl_overall_record_dict = {}

        if self.cl_algo:
            self.cl_wrapper.do_init(0, [train_dataloader], load_reg_params=configs.transfer)
            self.cl_wrapper.task_start_do()

        if self.stat_grad:
            _, _, _ = self.evaluate(eval_dataset=self.eval_dataset,
                                    verbose=False, reg_args=self.main_reg_args, global_iter=0)

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            N = len(epoch_iterator)
            for step, inputs in enumerate(epoch_iterator):
                if configs.debug:
                    if step == 10:
                        break
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer, reg_explanations, configs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        if self.cl_algo in ['ewc','lsp']:
                            try:
                                optimizer.step(reg_params=self.model.reg_params)
                            except TypeError:
                                logger.warning('error occurred when passing reg_params')
                                optimizer.step()
                        else:
                            optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if self.is_local_master():
                        if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
                                global_step == 1 and self.args.logging_first_step
                        ) or (step + 1) == len(epoch_iterator) or (step + 1) == self.epoch_len:
                            is_epoch_end = (step + 1) == len(epoch_iterator) or (step + 1) == self.epoch_len
                            logs = {}
                            if self.args.evaluate_during_training:
                                # results = self.evaluate()

                                # evaluate on the "eval" set (dev set)
                                results, raw_results, raw_inputs = self.evaluate(eval_dataset=self.eval_dataset,
                                                                                 verbose=True, reg_args=self.main_reg_args,
                                                                                 global_iter=global_step)
                                dev_filename = write_verbose_eval_results(self.args, raw_inputs, raw_results, self.eval_dataset, 'dev',
                                                           self.tokenizer, epoch=epoch, global_iter=global_step)

                                for key, value in results.items():
                                    eval_key = "eval_{}".format(key)
                                    logs[eval_key] = value
                                results_test, raw_results_test, raw_inputs_test = \
                                    self.evaluate(eval_dataset=self.test_dataset, verbose=True, reg_args=self.main_reg_args, global_iter=global_step)

                                test_filename = write_verbose_eval_results(self.args, raw_inputs_test, raw_results_test,
                                                           self.test_dataset,'test', self.tokenizer, epoch=epoch, global_iter=global_step)

                                if is_epoch_end:
                                    verbose_metrics = compute_metrics_from_csv(
                                        dev_csv_path=dev_filename,
                                        test_csv_path=test_filename, # not used
                                        task=self.full_configs.task_name,
                                        tokenizer=self.tokenizer,
                                        eer=self.save_eer,
                                        adv_fdcl=self.main_reg_args.adv_debias and self.full_configs.task_name == 'fdcl',
                                        data_dir=self.full_configs.data_dir,
                                        split='dev'
                                    )

                                    update_crit_metrics(main_record_dict, verbose_metrics, self.full_configs.task_name)

                                # if mtl, then evaluate over all the tasks
                                if self.mtl:
                                    dev_filenames = []
                                    for mtl_eval_task, mtl_eval_dataset, mtl_eval_metrics_func, mtl_reg_args in zip(
                                            self.mtl_args.mtl_tasks,
                                            self.all_eval_datasets[1:],
                                            self.mtl_args.mtl_metrics,
                                            self.mtl_args.mtl_reg_args):
                                        results_task, raw_results_task, raw_inputs_task = \
                                            self.evaluate(eval_dataset=mtl_eval_dataset,
                                                          eval_metrics_func=mtl_eval_metrics_func,
                                                          reg_args=mtl_reg_args, verbose=True)
                                        dev_filename = write_verbose_eval_results(self.args, raw_inputs_task.base_dataset, raw_results_task,
                                                                   mtl_eval_dataset, 'dev', self.tokenizer,
                                                                   epoch=epoch, global_iter=global_step)
                                        dev_filenames.append(dev_filename)
                                        for key, value in results_task.items():
                                            eval_key = "eval_{}".format(key)
                                            if eval_key in logs and type(logs[eval_key]) is not list:
                                                logs[eval_key] = [logs[eval_key]]
                                            if eval_key not in logs:
                                                logs[eval_key] = []
                                            logs[eval_key].append(value)
                                    for key in list(logs.keys()):
                                        if type(logs[key]) == list:
                                            logs[key + '_raw'] = logs[key]
                                            logs[key] = np.mean(logs[key])
                                    for i, (mtl_test_task, mtl_test_dataset, mtl_test_metrics_func, mtl_reg_args, mtl_task_args) in enumerate(zip(
                                            self.mtl_args.mtl_tasks,
                                            self.all_test_datasets[1:],
                                            self.mtl_args.mtl_metrics,
                                            self.mtl_args.mtl_reg_args,
                                            self.mtl_args.mtl_task_args)):
                                        results_task, raw_results_task, raw_inputs_task = \
                                            self.evaluate(eval_dataset=mtl_test_dataset,
                                                          eval_metrics_func=mtl_test_metrics_func,
                                                          reg_args=mtl_reg_args, verbose=True)
                                        test_filename = write_verbose_eval_results(self.args, raw_inputs_task.base_dataset, raw_results_task,
                                                                   mtl_test_dataset, 'test', self.tokenizer,
                                                                   epoch=epoch, global_iter=global_step)
                                        dev_filename = dev_filenames[i]
                                        if is_epoch_end:
                                            verbose_metrics_task = compute_metrics_from_csv(
                                                dev_csv_path=dev_filename,
                                                test_csv_path=test_filename,  # not used
                                                task=mtl_test_task,
                                                tokenizer=self.tokenizer,
                                                eer=self.save_eer,
                                                adv_fdcl=mtl_reg_args.adv_debias and mtl_test_task == 'fdcl',
                                                data_dir=mtl_task_args.mtl_data_dir,
                                                split='dev'
                                            )

                                            update_crit_metrics(mtl_record_dicts[mtl_test_task], verbose_metrics_task, mtl_test_task)
                                if is_epoch_end and not self.no_main_objective:
                                    best_mets = update_mtl_crit_metrics(mtl_overall_record_dict, [main_record_dict] + list(mtl_record_dicts.values()))

                                    for met in best_mets:
                                        output_dir = os.path.join(self.args.output_dir, 'best_{}'.format(met))
                                        self.save_model(output_dir)
                                        with open(os.path.join(output_dir,'epoch_step.txt'), 'w') as wf:
                                            wf.write('epoch\t{}\nglobal_step\t{}'.format(epoch, global_step))



                            loss_scalar = (tr_loss - logging_loss) / (self.args.logging_steps + 1e-10)
                            learning_rate_scalar = scheduler.get_last_lr()[0]
                            logs["learning_rate"] = learning_rate_scalar
                            logs["loss"] = loss_scalar
                            logger.info(str(logs))
                            logging_loss = tr_loss

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)
                            if is_wandb_available():
                                wandb.log(logs, step=global_step)

                            epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))

                            crit_value = logs['eval_' + criterion]
                            if criterion in ['loss']:
                                crit_value *= -1
                            logger.info('Early stop criterion: {}'.format(crit_value))
                            if crit_value <= best_result:
                                patience -= 1
                                epoch_iterator.write('Early stop patience %f' % patience)
                            else:
                                best_result = crit_value
                                patience = configs.early_stop

                                # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                # In all cases (even distributed/parallel), self.model is always a reference
                                # to the model we want to save.
                                if hasattr(model, "module"):
                                    assert model.module is self.model
                                else:
                                    assert model is self.model
                                # Save model checkpoint
                                output_dir = os.path.join(self.args.output_dir)

                                self.save_model(output_dir)
                                self._rotate_checkpoints()

                                ckpt_info = {'epoch': epoch, 'step': step, 'global_step': global_step}
                                with open(os.path.join(output_dir, 'ckpt_info.json'), 'w') as wf:
                                    json.dump(ckpt_info, wf)

                                # skip saving these checkpoints
                                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                                # logger.info("Saving optimizer and scheduler states to %s", output_dir)
                            if is_epoch_end and self.save_all:
                                output_dir = os.path.join(self.args.output_dir, 'epoch_{}'.format(epoch))
                                self.save_model(output_dir)

                            if patience <= 0:
                                break

                if patience <= 0 or (self.args.max_steps > 0 and global_step > self.args.max_steps):
                    epoch_iterator.close()
                    break
            if patience <= 0 or (self.args.max_steps > 0 and global_step > self.args.max_steps):
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        if self.cl_algo and not self.transfer:
            self.cl_wrapper.task_end_do()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
            reg_explanations=False, configs=None
    ) -> float:
        model.train()

        task_ids = None
        if 'task_ids' in inputs:
            task_ids = inputs.pop('task_ids').cpu().tolist()

        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        # currently head id is the same within batch
        outputs = model(**inputs, head_id=task_ids[0] if task_ids else None,
                        no_main_objective=self.no_main_objective)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()


        if (reg_explanations and (task_ids is None or task_ids[0] == 0)):
            explainer = self.explainer
            reg_loss, reg_cnt = explainer.compute_explanation_loss(
                inputs['input_ids'],
                inputs['attention_mask'],
                None,
                inputs['labels'],
                do_backprop=True,
            )
        elif task_ids is not None and self.mtl_args.get_targs_by_id(task_id=task_ids[0]).mtl_reg_args.reg_explanations:
            explainer = self.mtl_explainers[task_ids[0]]
            reg_loss, reg_cnt = explainer.compute_explanation_loss(
                inputs['input_ids'],
                inputs['attention_mask'],
                None,
                inputs['labels'],
                do_backprop=True,
            )


        return loss.item()

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the master process.
        """
        if self.is_world_master():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
            self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None, verbose=False,
            eval_metrics=None, eval_metrics_func=None, reg_args = None, global_iter=None
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation", eval_metrics=eval_metrics,
                                       eval_metrics_func=eval_metrics_func, reg_args=reg_args, global_iter=global_iter)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        if verbose:
            return output.metrics, output, eval_dataset
        else:
            return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None,
            eval_metrics: Optional[Dict] = None, eval_metrics_func=None, reg_args = None, global_iter=None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        model.to(self.args.device)

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", dataloader.batch_size)
        eval_losses: List[float] = []

        eval_adv_losses: List[float] = []  # storing adversarial classification loss
        eval_adv_preds, eval_adv_gts = [], []

        preds: np.ndarray = None
        label_ids: np.ndarray = None
        model.eval()

        total_grad_norm = 0
        total_reg_cnt, total_reg_loss = 0, 0

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])

            task_ids = None
            if 'task_ids' in inputs:
                task_ids = inputs.pop('task_ids').cpu().tolist()
                # assume all task ids are same in a batch

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            if not self.stat_grad:
                with torch.no_grad():
                    outputs = model(**inputs, head_id=task_ids[0] if task_ids else None)
                    if has_labels:
                        step_eval_loss, logits = outputs[:2]
                        eval_losses += [step_eval_loss.mean().item()]
                    else:
                        logits = outputs[0]
            else:
                outputs = model(**inputs, head_id=task_ids[0] if task_ids else None)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if self.stat_grad and self.stat_grad_method == 'expl':
                explainer = self.explainer
                reg_loss, reg_cnt = explainer.compute_explanation_loss(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    None,
                    inputs['labels'],
                    do_backprop=True,
                    importance_target=self.full_configs.importance_target
                )
                # compute grad
                norm = 0
                for name, p in model.named_parameters():
                    if 'classifier' not in name and p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        norm += param_norm.item() ** 2
                total_grad_norm += norm ** 0.5
                total_reg_cnt += reg_cnt
                total_reg_loss += reg_loss ** 0.5
                self.model.zero_grad()


            if reg_args is not None and reg_args.adv_debias or self.stat_grad_method == 'adv':
                extra_ret_dict = outputs[-1]
                attr = inputs['attr']

                adv_probs, adv_loss = extra_ret_dict['attr_probs'], extra_ret_dict['adv_loss']
                _, adv_preds = adv_probs.max(-1)  # [B]

                eval_adv_preds.extend(adv_preds.detach().cpu().numpy().tolist())
                eval_adv_gts.extend(attr.cpu().numpy().tolist())

                if adv_loss is not None:
                    eval_adv_losses.append(adv_loss.item())

                if self.stat_grad and self.stat_grad_method == 'adv':
                    self.model.zero_grad()
                    adv_loss.backward()
                    norm = 0
                    for name, p in model.named_parameters():
                        if 'classifier' not in name and 'adv' not in name and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            norm += param_norm.item() ** 2
                    total_grad_norm += norm ** 0.5
                    total_reg_cnt += 1
                    total_reg_loss += adv_loss
                    self.model.zero_grad()

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach().cpu().numpy()
                    else:
                        label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            preds = xm.mesh_reduce("eval_preds", preds, np.concatenate)
            label_ids = xm.mesh_reduce("eval_out_label_ids", label_ids, np.concatenate)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            if eval_metrics is None:
                if eval_metrics_func is not None:
                    metrics = eval_metrics_func(EvalPrediction(predictions=preds, label_ids=label_ids))
                else:
                    metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            else:
                metrics = eval_metrics
        else:
            metrics = {}

        # add metrics for adversarial debiasing
        if reg_args and reg_args.adv_debias:
            eval_adv_gts, eval_adv_preds = np.array(eval_adv_gts), np.array(eval_adv_preds)
            metrics['adv_f1'] = f1_score(eval_adv_gts[eval_adv_gts!=-1], eval_adv_preds[eval_adv_gts!=-1])
            metrics['adv_loss'] = np.mean(eval_adv_losses)

        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        if self.stat_grad:
            total_grad_norm /= 1e-10 + total_reg_cnt
            total_reg_loss /= 1e-10 + total_reg_cnt
            reg_info = 'Global iter: {}\tRegularizer graident: {}\tRegularizer loss: {}\timportance target: {}'.\
                format(global_iter, total_grad_norm, total_reg_loss, self.full_configs.importance_target)
            logger.info(reg_info)
            self.wf_stat_grad.write(reg_info + '\n')

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
