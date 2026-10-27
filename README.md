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
# CPU-only installation. For NVIDIA GPUs, see the separate instructions below.
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

### NVIDIA GPU setup (Linux)

The default `requirements_theseus.txt` installs CPU-only TensorFlow.
For NVIDIA GPUs, use the separate `requirements_theseus_gpu.txt` and do not
install both requirements files in the same environment. Python 3.11 is
recommended for the GPU setup below.

```bash
conda create -n SRN-GPU python=3.11 -y
conda activate SRN-GPU
python -m pip install --upgrade pip
python -m pip install -r requirements_theseus_gpu.txt
```

To verify CUDA installation, run:

```bash
python -c "import tensorflow as tf; print('CUDA build:', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**RTX 50-series / TensorFlow 2.21.0:** TensorFlow 2.21.0 has a confirmed
[cuSOLVER search-path issue](https://github.com/tensorflow/tensorflow/issues/113541).
Even with CUDA libraries installed, `GPUs: []` may appear. The following
workaround is scoped to the active shell and does not modify system CUDA:

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, site; p=os.path.join(site.getsitepackages()[0], "nvidia", "cusolver", "lib"); old=os.environ.get("LD_LIBRARY_PATH", ""); print(":".join([p]+[x for x in old.split(":") if x]))')"
```

NVIDIA Blackwell GPUs (including RTX 5070, compute capability 12.0a) also
require a sufficiently recent CUDA PTX compiler. If training logs report
`Used ptxas at /usr/bin/ptxas` with CUDA 12.0, prioritize the
CUDA 12.9 compiler included in the environment:

```bash
export PATH="$(python -c 'import site, pathlib; print(pathlib.Path(site.getsitepackages()[0]) / "nvidia/cuda_nvcc/bin")'):$PATH"
which ptxas
ptxas --version
```

Verify a real GPU operation and XLA compilation before starting a long run:

```bash
python -c "import tensorflow as tf; f=tf.function(lambda x: tf.matmul(x,x), jit_compile=True); x=tf.ones([64,64]); print('Device:', f(x).device)"
```

The result should contain `GPU:0`. On RTX 50-series GPUs, TensorFlow
may warn that kernels will be JIT-compiled from PTX. Successful execution on
`GPU:0` is the important check. The two `export` commands apply only to
the current terminal session. The system-installed NVIDIA driver and CUDA
toolkit do not need to be replaced.

### Download GloVe from Zenodo

The [Zenodo GloVe 6B mirror](https://zenodo.org/records/4925376)
provides the same Stanford GloVe 6B vocabulary in separate dimensions.
**Download `glove.6B.50d.zip` (58.9 MB)**, not the 100d, 200d, or 300d
archives. SRN's original question and relation encoders use 50-dimensional
vectors.

The Zenodo archive contains **binary Word2Vec-format vectors**, not the
plain-text GloVe file expected by `components/Embedder.py`. The extracted
file has **no `.bin` extension** (`glove.6B.50d`). Convert it once before
running SRN. From the repository root:

```bash
# Install the optional conversion dependency.
python -m pip install gensim

# Replace this filename if you saved the ZIP elsewhere.
python -m zipfile -e glove.6B.50d.zip datasets/embeddings

# Zenodo stores an extensionless binary file named glove.6B.50d.
# Convert that specific file to the text format expected by SRN.
python -c "from gensim.models import KeyedVectors; KeyedVectors.load_word2vec_format('datasets/embeddings/glove.6B.50d', binary=True).save_word2vec_format('datasets/embeddings/glove.6B.50d.txt', binary=False)"

# Optional sanity check: the first line should read '400000 50'.
head -n 1 datasets/embeddings/glove.6B.50d.txt
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

### Live training and evaluation progress

Interactive terminals display `tqdm` progress bars for every training epoch,
development evaluation, and final dev/test evaluation. Training bars show the
number of questions processed, elapsed time, throughput, estimated time remaining,
running loss, and Hits@1. The bar refreshes every 15 seconds by default, even
if a single query is taking a long time (for example, first-run CUDA compilation).

When output is redirected to a file or the terminal is non-interactive,
SRN instead prints plain-text progress heartbeats. Checkpoint messages,
per-epoch metric summaries, and `metrics.csv` are retained in both modes.

Set the refresh frequency without changing model training:

```bash
python theseus_experiment.py --dataset kinship --hops 3 --mode train \
  --glove datasets/embeddings/glove.6B.50d.txt \
  --relation-embeddings saved_models/theseus_kinship_single_3/transe_relations.npz \
  --progress-seconds 10
```

Set `--progress-style tqdm` to force bars, or `--progress-style plain` to
force log-friendly text. The default `--progress-style auto` uses bars only
when stderr is a terminal. `--progress-seconds 0` disables background
refreshes but keeps per-query bars or stage-start/end logs. ETA is approximate,
especially during first-run CUDA PTX compilation. `metrics.csv` records
`train_seconds`, `dev_seconds` and `epoch_seconds` in addition to
accuracy and loss. Progress instrumentation does not change the loss,
optimizer, rollout, reward or evaluation policy.

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
