# On Transferability of Bias Mitigation Effects in Language Model Fine-Tuning

## Requirement
See requirement.txt. We use PyTorch 1.4 and CUDA 10.1 in our experiments.

## Running experiments

### Training an upstream model
```shell script
task=gab # choices are gab, ws (stormfront), fdcl, dwmw, biasbios
base= # the root of the directory where you store models
exp_name= 
# the output dir would be <base>/<exp_name>/<seed>

# explanation regularization bias mitigation on GHC
python scripts/upstream.py --lr 1e-5 --seed 0 21 42 --task gab --name ${exp_name} --do_reg --base runs/${base} --extra "##reg_strength 0.03 ##neutral_words_file datasets/identity_gab.csv" --epoch 5

# vanilla model on GHC
python scripts/upstream.py --lr 1e-5 --seed 0 21 42 --task gab --name ${exp_name} --base runs/${base} --epoch 5

# mtl on GHC, FDCL, and biasbios
base= # the root of the directory where you store models
task=gab # the "main" task. Other tasks should can specified in "mtl args"
exp_name= 

python scripts/upstream.py --lr 1e-5 --seed 0 21 42 --task gab --name ${exp_name} --do_reg --extra "##reg_strength 0.03 ##neutral_words_file datasets/identity_gab.csv ##adv_lr_scale 100.0 ##logging_steps 0" --base runs-0711/three --epoch 8 --es 1000 --mtl --mtl_args "{'mtl_data_dir': 'datasets/FDCL18', 'mtl_task_name': 'fdcl', 'mtl_reg_args': {'reg_explanations': 1, 'reg_strength': 0.1, 'neutral_words_file': 'datasets/aae_words.csv'}}" "{'mtl_data_dir': 'datasets/biasbios', 'mtl_task_name': 'biasbios', 'mtl_reg_args': {'adv_debias': 1, 'adv_objective': 'adv_ce', 'adv_strength': 1.0, 'adv_grad_rev_strength': 1.0}}"

```
### Training a downstream model
```shell script
task=ws
exp_name=
source_path= # e.g. 'runs/roberta-base-vanilla/'
base=
python scripts/run_transfer.py --lr 1e-5 5e-6 --seed 0 21 42 --task ${task} --name ${exp_name}  --source ${source_path} --source_name ${source_name} --base ${base} --epoch 5 --es 1000 --extra "##load_epoch best_f1"

```


## Datasets included in this repository
- [Gab Hate Corpus (GHC)](https://osf.io/edua3/) released under CC-By Attribution 4.0 International License. 
version can be found in this repository.
- [Stormfront (Stf.)](https://github.com/Vicomtech/hate-speech-dataset) released under Creative Commons Attribution-ShareAlike 3.0 Spain
License. A preprocessed version can be found in this repository.
- [DWMW](https://github.com/t-davidson/hate-speech-and-offensive-language) released under MIT license and can be found in this directory.
- [iptts77k](https://github.com/conversationai/unintended-ml-bias-analysis) released under Apache License 2.0. 

If you use any of the datasets above, please consider citing the original work.

## Datasets not directly included in this repository
- [FDCL](https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN)
- [BiasBios](https://github.com/microsoft/biosbias)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) 
- [NYT corpus](https://catalog.ldc.upenn.edu/LDC2008T19) 

Similarly, if you use any of the datasets above, please consider citing the original work.

