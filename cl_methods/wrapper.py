from .regularizers import *
from .utils import WrapModel
from .parallel import DataParallelModel

class CLWrapper:
    def __init__(self, model, output_dir, method='ewc', explainer=None, expl_importance=False,
                 src_dir=None):
        self.model = model
        self.method = method
        self.parallel_model, self.regularizer = None, None
        self.output_dir = output_dir
        self.explainer = explainer
        self.expl_importance = expl_importance
        self.src_dir = src_dir

    def do_init(self, task, data_loaders, load_reg_params):
        self.parallel_model = DataParallelModel(WrapModel(self.model))
        self.regularizer = EWC(self.model, parallel_model=self.parallel_model, task=task, dataloaders=data_loaders,
                               output_dir=self.output_dir, src_dir=self.src_dir)
        if load_reg_params:
            if self.method == 'ewc':
                self.regularizer.load_reg_params(('classifier',))
            elif self.method == 'lsp':
                initialize_reg_params(self.model,freeze_layers=['classifier'], init_value=1)

    def task_start_do(self):
        self.regularizer.task_start_do(freeze_layers=['classifier'])

    def task_end_do(self):
        self.regularizer.task_end_do(explainer=self.explainer, expl_importance=self.expl_importance)