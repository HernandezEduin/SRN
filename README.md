# Multi-Relation-QA-over-KG
This is an implementation of the stepwise-reasoning network described in this [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371812).


Run all commands from the directory containing this README file. Code is ran with python 3.5 or above.

## Installing Libraries
Use [pip](https://pip.pypa.io/en/stable/) as a package manager to install the required libraries.

**If you only wish to see the model results:**
 
```bash
pip install matplotlib
```

**If you wish to train the model:**

```bash
pip install -r requirements.txt
```

## Visualizing the model results

```bash
python plot.py
```

Training and Validation results are displayed in separate plots.

## Training the model

### Download the dataset
A pre-trained embedding is used to embed the entities and relations used in our knowledge graph.
Due to the size of the datasets, these are not submitted, but they are required to run the model.

To download the dataset, follow the steps below.

1. Follow this [link](http://139.129.163.161/index/toolkits#pretrained-embeddings)
2. Download the Freebase dataset. (Submit the license agreement as required)
3. Unzip the file.
4. Save the Freebase folder into the datasets subfolder i.e. (/datasets/Freebase/)

### Train the model
 
```bash 
python experiment.py
```
A stepwise reasoning network and an ablated version of the network without the perceptron layer is trained using the PathQuestion dataset.

Results will be stored in the subfolder /saved_models/ in csv format.

## THESEUS: Kinship and MQuAKE-ST

An additional entry point, `theseus_experiment.py`, supports the official
[Kinship](https://huggingface.co/datasets/HalcyonSolutions/Kinship) and
[MQuAKE-ST](https://huggingface.co/datasets/HalcyonSolutions/MQuAKE-ST) releases
from [Theseus in the Graph](https://arxiv.org/abs/2609.14528). Run the following
commands from the repository root with Python 3.10+.

```bash
python -m pip install -r requirements_theseus.txt

# Download Kinship's original navigation KG and official two-hop QA split.
python theseus_experiment.py --dataset kinship --hops 2 --mode download

# Independently pretrain 50d TransE relation vectors on the navigation KG.
python prepare_embeddings.py \
  --graph datasets/theseus/kinship/kg/orig/triplets.txt \
  --output saved_models/theseus_kinship_single_2/transe_relations.npz

# Download the pretrained GloVe 6B, 50-dimensional text file separately,
# then train SRN using *only answers* as QA supervision.
python theseus_experiment.py --dataset kinship --hops 2 --mode train \
  --glove datasets/embeddings/glove.6B.50d.txt

# Evaluate an existing checkpoint without retraining.
python theseus_experiment.py --dataset kinship --hops 2 --mode evaluate \
  --glove datasets/embeddings/glove.6B.50d.txt
```

### Download GloVe from Zenodo

The [Zenodo GloVe 6B mirror](https://zenodo.org/records/4925376)
provides the same Stanford GloVe 6B vocabulary in separate dimensions.
**Download `glove.6B.50d.zip` (58.9 MB)**, not the 100d, 200d, or 300d
archives. SRN's original question and relation encoders use 50-dimensional
vectors.

The Zenodo archive contains **binary Word2Vec-format vectors**, not the
plain-text GloVe file expected by `components/Embedder.py`. Convert it
once before running SRN. From the repository root:

```bash
# Install the optional conversion dependency.
python -m pip install gensim

# Replace this filename if you saved the ZIP elsewhere.
python -m zipfile -e glove.6B.50d.zip datasets/embeddings

# Find the extracted .bin file and convert it to text.
python -c "from pathlib import Path; from gensim.models import KeyedVectors; src = next(Path('datasets/embeddings').rglob('*.bin')); KeyedVectors.load_word2vec_format(str(src), binary=True).save_word2vec_format('datasets/embeddings/glove.6B.50d.txt', binary=False)"
```

The resulting `datasets/embeddings/glove.6B.50d.txt` works with
`--glove datasets/embeddings/glove.6B.50d.txt`. The text export has
a Word2Vec header (`400000 50`), which the SRN text loader skips.
These vectors initialize **word embeddings**, not KG entity/relation
embeddings. The separate TransE preparation step is still required.
The original [Stanford GloVe distribution](https://nlp.stanford.edu/projects/glove/)
is an alternative source for the text file if reachable.

The original SRN code relied on a Freebase-specific GloVe + TransE package,
which cannot be used directly for the Kinship or Wikidata relation vocabulary.
`prepare_embeddings.py` supplies an **optional independent TransE pretraining**
implementation. It does not use QA labels or evidence paths. You may supply
your own pretrained 50-dimensional TransE vectors with
`--relation-embeddings /path/to/vectors.npz`. The NPZ file contains arrays
`relations` (strings) and `vectors` (number of relations × 50, float32).
Embedding provenance and pretraining hyperparameters must be reported in
comparisons. This adapter is not a claim to reproduce the authors' unpublished
pretrained Freebase vectors.

### Available evaluations

| Dataset | `--hops` | `--answer-mode` | Navigation KG |
| --- | --- | --- | --- |
| Kinship | `2`, `3`, `n` | `single` | `kg/orig/triplets.txt` |
| MQuAKE-ST | `2`, `3`, `4`, `n` | `single` or `multi` | `kg/triplets.txt` |

For example, use the following for MQuAKE-ST multi-answer three-hop QA:

```bash
python theseus_experiment.py --dataset mquake-st --hops 3 \
  --answer-mode multi --mode download
python prepare_embeddings.py \
  --graph datasets/theseus/mquake-st/kg/triplets.txt \
  --output saved_models/theseus_mquake-st_multi_3/transe_relations.npz
python theseus_experiment.py --dataset mquake-st --hops 3 \
  --answer-mode multi --glove datasets/embeddings/glove.6B.50d.txt
```

The loader uses the **official `SplitLabel`**, explicit `Source-Entity`, and
encoded `Answer-Entity` fields. The static graph is directed. Parallel
relations are preserved, and no graph inverses or extra gender edges are
introduced. Gold `Paths`, `Paths-Label`, and relation chains are not consumed
by the model. In multi-answer mode, reaching **any** released answer is correct.

To avoid giving the agent a gold hop count at evaluation, we recommend fixed
hop subsets (`--hops 2`, `3`, or `4`). Mixed `--hops n` is supported only with
the explicit `--oracle-hops` acknowledgment because the original fixed-horizon
SRN has no learned STOP action. Results using oracle horizons should **not**
be compared directly with agents that infer their own stopping behavior.

### Training and results

The model retains the original SRN's 50d GloVe inputs, frozen TransE relation
vectors, BiGRU question encoder, hop-specific SLP, relation attention,
semantic perceptron, GRU history, answer-level potential-based reward shaping,
and REINFORCE. It does **not** introduce evidence-path supervision or a new
loss. The executable path repairs issues in the archived third-party code,
including malformed attention tensor shapes, GRU history handling, use of the
policy-gradient loss, and label-dependent action pruning at evaluation.
Its beam search ranks by learned relation-policy scores, never gold answers.

Outputs go to `saved_models/theseus_<dataset>_<mode>_<hops>/` and include
`metrics.csv`, model checkpoints, and `dev_predictions.jsonl` and
`test_predictions.jsonl` with explicit trajectories. `--seed` controls
NumPy, TensorFlow, and training order. Use `--limit N` **only for debugging**.
The original `experiment.py` and historical PathQuestion results are retained
as references, but the legacy environment is not guaranteed to run unchanged.

Run adapter tests using:

```bash
python -m unittest -v test_theseus_data.py
```
