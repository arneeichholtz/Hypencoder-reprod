# Hypencoder Reproduction
This repository was adapted from the repo for the original paper, "Hypencoder: Hypernetworks for Information Retrieval" by Killingback et al. Link: https://arxiv.org/pdf/2502.05364

The core code, data, and models are now available. This means you can train your own Hypencoder, use a pre-trained Hypencoder off-the-shelf, and reproduce the major results from the paper exactly.

![main_image](imgs/Figure_1.jpg)

<h4 align="center">
    <p>
        <a href=#installation>Installation</a> |
        <a href=#quick-start>Quick Start</a> |
        <a href=#models>Models</a> |
        <a href=#data>Data</a> |
        <a href=#artifacts>Artifacts</a> |
        <a href=#extensions>Extensions</a> |
        <a href=#training>Training</a> |
        <a href="#cite">Citation</a>
    <p>
</h4>

## Installation
### Copy the Repo
```
git clone [github link]
```

### Make a conda environment and install the requirements
```
conda create --name hype python=3.10 -y
source activate hype
pip install -r requirements.txt
```

### Required Libraries
The core libraries required are:
- `torch`
- `transformers`

with just the core libraries you can use Hypencoder to create q-nets and
document embeddings.

To use the code for encoding and retrieval the following additional libraries
are required:
- `fire`
- `tqdm`
- `ir_datasets`
- `jsonlines`
- `docarray`
- `numpy`
- `ir_measures`

To train a model you will need:
- `fire`
- `omegaconf`
- `datasets`

To use Faiss you will need an additional library, this can be `pip` installed in the active environment:
- `faiss-gpu`

## Quick Start
#### Using the pretrained Hypencoders as stand-alone models
```python
from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderDualEncoder, TextEncoder
from transformers import AutoTokenizer

dual_encoder = HypencoderDualEncoder.from_pretrained("jfkback/hypencoder.6_layer")
tokenizer = AutoTokenizer.from_pretrained("jfkback/hypencoder.6_layer")

query_encoder: Hypencoder = dual_encoder.query_encoder
passage_encoder: TextEncoder = dual_encoder.passage_encoder

queries = [
    "how many states are there in india",
    "when do concussion symptoms appear",
]

passages = [
    "India has 28 states and 8 union territories.",
    "Concussion symptoms can appear immediately or up to 72 hours after the injury.",
]

query_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
passage_inputs = tokenizer(passages, return_tensors="pt", padding=True, truncation=True)

q_nets = query_encoder(input_ids=query_inputs["input_ids"], attention_mask=query_inputs["attention_mask"]).representation
passage_embeddings = passage_encoder(input_ids=passage_inputs["input_ids"], attention_mask=passage_inputs["attention_mask"]).representation

# The passage_embeddings has shape (2, 768), but the q_nets expect the shape
# (num_queries, num_items_per_query, input_hidden_size) so we need to reshape
# the passage_embeddings.

# In the simple case where each q_net only takes one passage, we can just
# reshape the passage_embeddings to (num_queries, 1, input_hidden_size).
passage_embeddings_single = passage_embeddings.unsqueeze(1)
scores = q_nets(passage_embeddings_single)  # Shape (2, 1, 1)
# [
#    [[-12.1192]],
#    [[-13.5832]]
# ]

# In the case where each q_net takes both passages we can reshape the
# passage_embeddings to (num_queries, 2, input_hidden_size).
passage_embeddings_double = passage_embeddings.repeat(2, 1).reshape(2, 2, -1)
scores = q_nets(passage_embeddings_double)  # Shape (2, 2, 1)
# [
#    [[-12.1192], [-32.7046]],
#    [[-34.0934], [-13.5832]]
# ]
```

#### Encoding and Retrieving
If the queries and documents you want to retrieve exist as a dataset in the IR Dataset library no additional work is needed to encode and retrieve from the dataset. If the data is not a part of this library you will need two JSONL files for the documents and queries. These must have the format:
```
{"<id_key>": "afei1243", "<text_key>": "This is some text"}
...
```
where `<id_key>` and `<text_key>` can be any string and do not have to be the same for the document and query file.

##### Encoding
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$ENCODING_PATH \
--jsonl_path=path/to/documents.jsonl \
--item_id_key=<id_key> \
--item_text_key=<text_key>
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/encode.py --help`.

##### Retrieve
The values of `ENCODING_PATH` and `MODEL_NAME_OR_PATH` should be the same as
those used in the encoding step.
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--query_jsonl=path/to/queries.jsonl \
--do_eval=False \
--query_id_key=<id_key> \
--query_text_key=<text_key> \
--query_max_length=64 \
--top_k=1000
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/retrieve.py --help`.

##### Evaluation
Evaluation is done automatically when `hypencoder_cb/inference/retrieve.py` is called so long as `--do_eval=True`. If you are not using an IR Dataset you will need to provide the qrels with the argument `--qrel_json`. The qrels JSON should be in the format:
```
{
    "qid1": {
        "pid8": relevance_value (float),
        "pid65": relevance_value (float),
        ...
    }.
    "qid2": {
        ...
    },
    ...
}
```

#### Replications
For an overview of the exact `IR` datasets used in the original paper and our reproduction, see `replication_commands.md` in this repository.

#### Custom Q-Nets
In the paper we only looked at simple linear q-nets but in theory any type of neural network can be used. The code in this repository is flexible enough to support any q-net whose only learnable parameters can be expressed as a set of matrices and vectors. This should include almost every neural network.

To build a custom q-net you will need to make a new q-net converter similar to the existing one `RepeatedDenseBlockConverter`. This converter must have the following functions and properties:
1. `weight_shapes` should be a property which is a list of tuples indicating the size of the weight matrices.
2. `bias_shapes` should be a property which is a list of tuples indicating the size of the bias vectors.
3. `__call__` which takes three arguments `matrices`, `vectors`,  and `is_training`. See `RepeatedDenseBlockConverter` for details on the type of these arguments. This method should
return a callable object which excepts a torch tensor in the shape (num_queries, num_items_per_query, hidden_dim) and returns a tensor with the shape (num_queries, num_items_per_query, 1) which contains the relevance score for each query and associated item.

## Training
To train a model take a look at the training readme in `/train`.

## Models
We have uploaded the models from our experiments to Huggingface Hub. See quick start for more information on how to use these models and our paper for more information on how they were trained.
<center>

| Huggingface Repo | Number of Layers |
|:------------------:|:------------------:|
| [jfkback/hypencoder.2_layer](https://huggingface.co/jfkback/hypencoder.2_layer) |          2        |
| [jfkback/hypencoder.4_layer](https://huggingface.co/jfkback/hypencoder.4_layer) |          4        |
| [jfkback/hypencoder.6_layer](https://huggingface.co/jfkback/hypencoder.6_layer) |          6        |
| [jfkback/hypencoder.8_layer](https://huggingface.co/jfkback/hypencoder.8_layer) |          8        |
</center>

## Model checkpoints
Additional model checkpoints have been made available:  

[TOT Models](https://drive.google.com/drive/folders/1iMwgvoTbae9AY5IRXjf-UIwCLMGLaJL1?usp=drive_link)

[FollowIR Models](https://drive.google.com/drive/folders/13oGWmJCuQPe-xnMiiqi2Kk0qyguJLE87?usp=drive_link)

[BE-Baseline](https://drive.google.com/drive/folders/1iZVeZmP66e9NADY6UMUqq3SXfrnU1fqY?usp=drive_link)


## Data
The data of the original paper experiments is in the table below:
<center>

| Link | Description |
|:------------------:|------------------|
| [jfkback/hypencoder-msmarco-training-dataset](https://huggingface.co/datasets/jfkback/hypencoder-msmarco-training-dataset) | Main training data used to train all our Hypencoder models and BE-base |
</center>

The data for fine-tuning on hard tasks can be obtained through:
```
python script/load_data.py \
--data FollowIR_train \ #Options: ["FollowIR_train", "FollowIR_test", "TOT_train", "TOT_test", "DL_HARD_test"]
--dest data/followir/train

```

## Extensions
### Retrieve with Faiss
To use BE-Base with Faiss, there is a dedicated `encode` script that should be used:

```
python scripts/be_base/encode_bebase.py \
	--model_name_or_path=[path_to_be_base_checkpoint] \
	--ir_dataset_name="beir/hotpotqa/test" \
	--output_path=[encoded_items_folder/data].docs \
	--batch_size=64
```

And there is a specific script to handle retrieval using Faiss. Note that `faiss-gpu` should be installed in the enviroment prior to running. The script can be called as follows:

```
python scripts/be_base/retrieve_bebase_faiss.py \
	--model_name_or_path=[path_to_be_base_checkpoint] \
	--ir_dataset_name="beir/hotpotqa/test" \
	--encoded_docs_path=[encoded_items_folder/data].docs  \          # Same path for encoding
	--output_dir=[retrieval_output_folder] \
	--top_k=1000

```

This script uses the tokenizer with the checkpoint, encodes both documents and queries using the BE-base dual encoder, and reports standard IR metrics via `ir_measures`.

### Fine-tune Hypencoder Reproduction
```
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/finetune_FollowIR.yaml
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/finetune_TOT.yaml
```

### Fine-tune Alternative Encoders
```
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/contriever_freeze_encoder.yaml 
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/contriever_nofreeze_encoder.yaml

python hypencoder_cb/train/train.py hypencoder_cb/train/configs/tasb_freeze_encoder.yaml
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/tasb_nofreeze_encoder.yaml

python hypencoder_cb/train/train.py hypencoder_cb/train/configs/retro_freeze_encoder.yaml
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/retro_nofreeze_encoder.yaml
```

### Adversarial Attacks
Generate attacked queries: 
```
python scripts/adversarial_attack/attacks.py \
    --ir_dataset_name=$IR_DATASET_NAME \
    --attack_types all \
    --output_dir=$OUTPUT_DIR
```
Get retrieval results on attacked queries:
```
python hypencoder_cb/inference/retrieve_multiple.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODING_PATH \
    --base_data_dir=$BASE_DATA_DIR \
    --base_output_dir=$BASE_OUTPUT_DIR \
    --attack_types "['synonym', 'paraphrase', 'naturality', 'mispelling', 'ordering']" \
    --qrel_json=$QREL_JSON \
    --query_id_key="id" \
    --query_text_key="text" \
    --query_max_length=64
```
For combining the results of TREC 2019 and 2020 run:
```
ATTACK_TYPES=("mispelling" "naturality" "ordering" "paraphrase" "synonym")
# ATTACK_TYPES=("regular")

BASE_DIR="$HOME/hypencoder/retrieval_outputs"
SCRIPT_DIR="$HOME/hypencoder/scripts"

for ATTACK in "${ATTACK_TYPES[@]}"; do
  METRICS19="$BASE_DIR/adversarial/be_base/trec-dl-2019/${ATTACK}/metrics/per_query_metrics.json"
  METRICS20="$BASE_DIR/adversarial/be_base/trec-dl-2020/${ATTACK}/metrics/per_query_metrics.json"
  OUT_DIR="$BASE_DIR/adversarial/be_base/trec-dl-19-20/${ATTACK}/metrics"
  OUT_FILE="$OUT_DIR/aggregated_metrics.json"

  echo "Aggregating metrics for attack: ${ATTACK}"
  echo "  2019: $METRICS19"
  echo "  2020: $METRICS20"
  echo "  Out : $OUT_FILE"

  mkdir -p "$OUT_DIR"

  python "$SCRIPT_DIR/aggregate_trec_metrics.py" \
    --metrics1_path "$METRICS19" \
    --metrics2_path "$METRICS20" \
    --output_path "$OUT_FILE"
done
```
For generating the barplot (note: requires relative performance drops to be entered manually):
```
python scripts/plot_adversarial_attacks.py
```

## Artifacts
The artifacts from our experiments are in the table below:
<center>

| Link | Description |
|:------------------:|------------------|
| [hypencoder.6_layer.encoded_items](https://drive.google.com/drive/folders/1htoVx8fAVm-4ZfdssAXdw-_D-Kzs59dx?usp=sharing) | 6 layer Hypencoder embeddings for MSMARCO passage |
| [hypencoder.6_layer.neighbor_graph](https://drive.google.com/file/d/1EhKuGxaFI51DDSDqsoAwiYRs1IdZATrk/view?usp=sharing) | 6 Layer Hypencoder passage neighbor  graph for MSMARCO passages - needed for approximate search. |
| [Run Files](https://drive.google.com/drive/folders/1Q1U9Aa4bw_wK-EbAs9xCyUvBYxoUOVEG?usp=sharing) | All the run files for experiments |
</center>

The above artifacts are stored on Google Drive, if you want to download them without going through the UI you can, but I suggest looking at [gdown](https://github.com/wkentaro/gdown) or the Google Drive interface provided by [rclone](https://rclone.org/drive).

We have also uploaded all the run files for our experiments (FollowIR coming soon). They are a custom JSONL format, but they should be pretty straightforward to convert to any other format. We may also add standard TREC run files in future if there is interest.


## Citation
When using this repository, please cite the original paper:
```
@inproceedings{hypencoder,
    author = {Killingback, Julian and Zeng, Hansi and Zamani, Hamed},
    title = {Hypencoder: Hypernetworks for Information Retrieval},
    year = {2025},
    isbn = {9798400715921},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3726302.3729983},
    doi = {10.1145/3726302.3729983},
    booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {2372–2383},
    numpages = {12},
    keywords = {learning to rank, neural ranking models, retrieval models},
    location = {Padua, Italy},
    series = {SIGIR '25}
}
```
