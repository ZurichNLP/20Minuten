# 20 Minuten Dataset

## Environment setup

```bash
module load volta anaconda3 cuda/10.2 gcc/7.4.0 # on s3it only

conda create -n 20min python=3.8.5 -y
conda activate 20min
pip install -r requirements.txt
```

## 20 Minuten Data

To download the data, run

```bash

#TODO

```

For experimation, we provide a data handling script to load the dataset for a given set of tasks with Hugging Face datasets.

For this to work, you must set the paths to the local dataset in `path_config.json`.

## Experiments

If you are interested in reproducing our baseline experiments, we provide a few pointers below.

We recommend using symbolic links to recreate the directory structure described below.
This will avoid having to change these in the scripts when moving between machines.


```bash
├── 
├── logs
├── examples
├── plots
├── scripts # python scripts and jupyter notebooks for experiments and analyses
├── utils # dependency modules
├── resources # directory or symlink to directory containing large files
    ├── data
    ├── models 
    └── results
```

The `data` directory should contain the downloaded 20Minuten dataset.

Note, alternatively, you can just change any hardcoded paths to reflect your local setup.

## Data leakage

As discussed in the paper, train, test and validation splits are created randomly. 
However, we found evidence of data leakage between test and validation splits and mT5's pre-training corpus mC4. 
The results reported in the paper are therefore computed on filtered test/validation splits.

To reproduce these filtered splits, run:

```bash
python scripts/data_cleaning.py \
    -i resources/data/20_min_0_00_001 \
    -m resources/data/20min_articles_in_mc4.jsonl \
    -k title
```

## Model Conversion

To convert mT5 to a much smaller deT5 model, by throwing away unnecessary vocabulary items, we provide a Jupyter Notebook `convert_mt5_to_det5.ipynb`.

To collect relevant subwords, we used 1 million sentences from the German Newscrawl dataset (2019). This dataset is available [here](https://corpora.uni-leipzig.de/en?corpusId=deu_news_2019).

## Fine-tuning

We performed our experiments on a SLURM cluster. To reproduce, run

```bash
sbatch scripts/run_finetune.sh -b /path/to/repo -t task_name
```

NOTE: valid `task_names` include:
 - `title_generation`
 - `lead_generation`
 - `summary_generation` 
 - `caption_generation` 
 - `category_prediction` 
 - `reading_time_prediction` 
 - `multitask`

If working in an interactive session, you can execute jobs directly using the functions (e.g. `train_20min_lead`) declared in `experiments.sh` directly, e.g.

```
export CUDA_VISIBLE_DEVICES=X
source experiments.sh && train_20min_lead . det5-base
```

## Inference

Inference on the test set is done as part of fine-tuning. To re-run inference, e.g. on a new test split, use:

```bash
sbatch scripts/run_inference.sh
```

## Inspecting model generations

Once models have been trained for all six tasks and inference has been done, you can use the script `scripts/get_example_outputs.sh` to randomly select an article and all corresponding model outputs. e.g.

```bash
bash scripts/get_example_outputs.sh > examples/01.txt
```


## Corpus Issues / Bugs

We are constantly working on improving the quality of this corpus. To this end, we list some of the major issues that we've found and hope to fix in future versions.

- [ ] articles with embedded content rather than raw text should be removed 
    - e.g. https://www.20min.ch/de/story/wir-machen-nicht-auf-scheissegal-das-ist-unser-leben-899436050394
- [ ] hyperlinks in summaries are currently breaking sentences
    - e.g. https://www.20min.ch/de/story/wir-machen-nicht-auf-scheissegal-das-ist-unser-leben-899436050394
- [ ] some article titles appear in multiple splits with apparently different content
    - e.g. https://www.20min.ch/story/trump-unterzeichnet-sanktionsgesetz-gegen-china-185558269103 vs. https://www.20min.ch/story/trump-unterzeichnet-sanktionsgesetz-gegen-china-275663625810
- [ ] inconsistent use of html tags causes some articles to lose linebreaks.
    - e.g. 2021-08-03_6_350205291987.json: "QuarantäneEs gibt keine Quarantänepflicht nach der Einreise aus der Schweiz".


## Citation

```
TODO
```
