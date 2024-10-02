# Learning Real Bug Detectors
This is the official repository for the paper: [On Distribution Shift in Learning-based Bug Detectors](https://arxiv.org/abs/2204.10049).

## Setup
The code requires `python3` (we use `python3.9`) and some Python packages that can be installed via `pip install -r requirements.txt`. Make sure to add this repository to `PYTHONPATH`.

## Downloading Datasets and Models
We provide the following resources for download:
- Our datasets: [link](https://files.sri.inf.ethz.ch/icml22/dataset.tar.gz).
- Our fine-tuned models: [link](https://files.sri.inf.ethz.ch/icml22/fine-tuned.tar.gz).
- Pretrained models (converted from [CuBERT](https://github.com/google-research/google-research/tree/master/cubert), including the tokenizer vocabulary): [link](https://files.sri.inf.ethz.ch/icml22/pretrained.tar.gz).

After downloading and decompressing the above files, the directory structure should be organized as follows:
```
└──learning-real-bug-detector
│
└──dataset
│   
└──fine-tuned
│   
└──pretrained
```

## Running the Code
You can run the code via the scripts under the `scripts/` directory.

### Evaluation and Fine-tuning
Evaluation can be done with the command below, where `TASK_NAME` is the bug type (`var-misuse`, `wrong-binary-operator`, or `argument-swap`). MODEL_NAME is the name of the model (e.g., `model` if you use our fine-tuned models). Optionally, you can use the `--probs_file` to store the prediction results and use `calculate_ap.py` to compute average precision.
```
(scripts/) $ python eval.py --task TASK_NAME --model MODEL_NAME
```

Fine-tuning can be done with the command below, where `DATASET_NAME` can be `real`, `synthetic`, or `contrastive`. The paper describes a two-phase training scheme, first with `--dataset contrastive` and then with `--dataset real` (use `--pretrained` to continue from the previous checkpoint). Other fine-tuning parameters are defaulted to be the best parameters in the paper evaluation.
```
(scripts/) $ python fine-tune.py --task TASK_NAME --model MODEL_NAME --dataset DATASET_NAME
```

### Constructing Datasets from Scratch
If you are interested in constructing the datasets from scratch, you need to clone [eth_py150_open](https://github.com/google-research-datasets/eth_py150_open), download [py150_files](http://files.srl.inf.ethz.ch/data/py150_files.tar.gz), and install [near-duplicate-code-detector](https://github.com/microsoft/near-duplicate-code-detector). For `var-misuse` and `wrong-binary-operator`, the datasets constructed from `eth_py150` repositories have a sufficient amount of real bugs. For `argument-swap`, more repositories are needed to produce enough real bugs. The directory structure should be organized as follows:
```
└──learning-real-bug-detector
    │
    └──data
        │
        └──near-duplicate-code-detector
│
└──eth_py150_open
│   
└──py150_files
    │
    └──data
```

Then run the following commands:
```
(scripts/data_gen_real/) $ python clone_repos.py --in_file all_py150_repos.txt
(scripts/data_gen_real/) $ ./run_real_bugs_from_repo.sh TASK_NAME
(scripts/data_gen_synthetic/) $ python gen_jsontxt.py --task TASK_NAME
(scripts/data_gen_synthetic/) $ python clean_jsontxt.py --task TASK_NAME
(scripts/data_gen_real/) $ python split_real.py --task TASK_NAME
(scripts/data_gen_synthetic/) $ python filter_train_data.py --task TASK_NAME
(scripts/data_gen_synthetic/) $ python gen_synthetic_train_data.py --task TASK_NAME
(scripts/data_gen_synthetic/) $ python gen_synthetic_train_data.py --task TASK_NAME --contrastive
```
