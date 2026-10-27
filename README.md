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

# CPU-only installation. For NVIDIA GPUs, see the separate instructions below.
```bash
python -m pip install -r requirements_theseus.txt
```

# Download Kinship's original navigation KG and official n-hop QA split.
```bash
python theseus_experiment.py --dataset kinship --hops n --mode download
```

# Independently pretrain 50d TransE relation vectors on the navigation KG.
```bash
python prepare_embeddings.py \
  --graph datasets/theseus/kinship/kg/orig/triplets.txt \
  --output saved_models/kinship_transe/relations.npz \
  --dim 50 \
  --epochs 100
```

# Download the pretrained GloVe 6B, 50-dimensional text file separately,
# then train SRN using *only answers* as QA supervision.
```bash
python theseus_experiment.py --dataset kinship --hops 3 --mode train \
  --glove datasets/embeddings/glove.6B.50d.txt \
  --relation-embeddings saved_models/kinship_transe/relations.npz \
  --output saved_models/kinship_srn \
  --seed 42
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
archives. This THESEUS adapter currently uses 50-dimensional
vectors. The published SRN paper reports a 300-dimensional configuration.

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

### Verify GloVe vocabulary coverage

SRN checks GloVe coverage **before training or evaluation**, using the exact
lowercased tokens produced by `theseus_data.question_tokens` and consumed by
`Embedder.embed_word`. Like the UHop preprocessing diagnostic, it reports
coverage for both unique words and word occurrences. SRN additionally reports
these metrics separately for train, dev, and test, and lists the most frequent
out-of-vocabulary (OOV) tokens. No extra tokenizer, vocabulary expansion, or
training change is introduced.

For example, the startup log includes:

```text
GloVe vocabulary coverage (50d, SRN question tokens):
  all: unique <matched>/<total> (<percent>), occurrences <matched>/<total> (<percent>)
  train: ...
  dev: ...
  test: ...
  Excluded special tokens: {'<e>': <count>}
  Top GloVe OOV words (... total): ...
GloVe coverage saved to saved_models/.../glove_coverage.json
```

The full `glove_coverage.json` retains all OOV words and their occurrence
counts to support reproducibility. The `<e>` entity marker is intentionally
represented by a zero vector in this implementation and is excluded from
ordinary word coverage, although its occurrences are reported. **GloVe covers
question words only**. TransE relation vectors are checked separately for all
KG relation IDs.

The original SRN code relied on a Freebase-specific GloVe + TransE package,
which cannot be used directly for the Kinship or Wikidata relation vocabulary.
`prepare_embeddings.py` independently trains **50-dimensional TransE** using
only KG triples. No QA labels, questions, or reference paths are used.
It uses head/tail corruption, a TransE L1 distance score, and the link-prediction
logistic objective, following the relevant parts of
[MultiHopKG's KGE trainer](https://github.com/HernandezEduin/MultiHopKG/blob/master/kge_train.py).
This KGE pretraining is separate from SRN's original answer-reward REINFORCE
training, which is unchanged.

### TransE link-prediction validation and vector mapping

TransE training first makes reproducible, disjoint **KG triple** partitions
(default 80% train, 10% validation, 10% test). Every entity and relation must
still appear in the training partition. Model selection uses validation
**filtered MRR**. The held-out test partition is evaluated once after model
selection, reporting filtered MRR, mean rank, and Hits@1/3/10 for **both**
missing-head and missing-tail prediction. Filtering excludes other true KG
triples from the competing candidates. These KGE splits are independent of
the THESEUS **QA** splits.

Run from the repository root:

```bash
python prepare_embeddings.py \
  --graph datasets/theseus/kinship/kg/orig/triplets.txt \
  --output saved_models/theseus_kinship_single_3/transe_relations.npz \
  --epochs 100 --negatives 16 --eval-every 5 --seed 42 \
  --refit-full
```

`--refit-full` is optional. After selecting an epoch count and evaluating on
held-out triples, it initializes a new TransE model and trains on **all KG
triples** for that many epochs to export SRN's relation vectors. The held-out
scores in the JSON report describe the **validation-selected model before
refitting**, not the final full-graph model. Omit `--refit-full` if you prefer
to export the exact held-out-evaluated checkpoint.

The outputs next to `transe_relations.npz` are:

- `transe_relations.npz`: unchanged SRN-compatible `relations` and `vectors`
  arrays with shape `(n_relations, 50)`.
- `transe_relations.npz.entities.npz`: `entities` and `vectors` arrays with
  shape `(n_entities, 50)` for inspecting the KGE entity representations.
- `transe_relations.npz.training.csv`: per-epoch training loss and validation
  metrics at evaluation epochs.
- `transe_relations.npz.metrics.json`: reproducibility parameters, partition
  sizes, best validation epoch, held-out metrics, and export provenance.

**The mapping is stored inside each NPZ.** Specifically,
`relations[i]` is the exact KG relation identifier whose vector is
`vectors[i]`. `components/Embedder.py` reconstructs a dictionary with
`dict(zip(relations, vectors))`, so there is no dependence on an implicit
integer ordering across runs. The entity sidecar uses the same convention.
To inspect actual identifiers and the first coordinates:

```bash
python -c "import numpy as np; d=np.load('saved_models/theseus_kinship_single_3/transe_relations.npz',allow_pickle=False); print([(str(name), v[:5].tolist()) for name,v in zip(d['relations'],d['vectors'])])"
```

This mapping uses **KG identifiers** (for example relation labels in Kinship
or Wikidata PIDs in MQuAKE-ST), not GloVe word tokens. GloVe text embeddings
remain separate.

The trainer uses TensorFlow, so the existing GPU environment is sufficient.
By default it evaluates at most 256 held-out triples per validation/test split,
always ranking against **every graph entity**. Raise `--eval-samples` for
more accurate estimates, or set it to `0` for full evaluation, which can be
expensive on MQuAKE-ST. `--adversarial` enables self-adversarial negative
weighting. The selected epoch and pretraining settings should be reported when
comparing SRN baselines.

You may also supply externally pretrained 50-dimensional TransE vectors using
`--relation-embeddings /path/to/vectors.npz`, with the same two NPZ arrays.
Our independent training is not a claim to reproduce the original SRN
authors' unpublished Freebase embeddings.

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

### Published SRN reward and beam-search efficiency

The THESEUS adapter now matches the published SRN base reward in Section 2.4:
**only final-answer correctness** determines the episode's base reward.
When the complete trajectory reaches an answer, every decision receives the
positive base reward before potential-based shaping. An intermediate visit to
an answer followed by a wrong endpoint receives no positive **base** reward.
The potential shaping term remains
`gamma * phi(next_state) - phi(current_state)`, and the policy continues to use
on-policy REINFORCE without reference-path supervision.

Previous THESEUS SRN checkpoints were trained with an intermediate-answer
reward and should **not** be mixed into experiments using the corrected
objective. Use a new `--output` directory and retrain from scratch when
comparing results.

Evaluation keeps the same learned relation scores and global beam ordering,
but computes candidate relation scores in batches, advances GRU histories
only for branches surviving beam pruning, and omits unnecessary history
updates at the final step. This avoids a large number of tiny TensorFlow
operations during development/test evaluation. The result is intended to
preserve the exact inference decisions, aside from ordinary floating-point
roundoff. Actual speedup will depend on GPU, KG branching and beam width.

To reduce frequent expensive dev runs while retaining full evaluation on the
final epoch, set `--dev-every N`, for example:

```bash
python theseus_experiment.py --dataset kinship --hops 3 --mode train \
  --glove datasets/embeddings/glove.6B.50d.txt \
  --relation-embeddings saved_models/kinship_transe/relations.npz \
  --output saved_models/kinship_srn_paper_reward \
  --epochs 100 --dev-every 5 --seed 42
```

The default `--dev-every 1` preserves development evaluation every epoch.
Skipped evaluations have blank `dev_hits1` and zero `dev_seconds` in
`metrics.csv`. Checkpoints are still saved every epoch, and full final
dev/test evaluation still runs after training. Development and test use the
existing `--beam-size` setting (default 32).

### Question-conditioning diagnostic

To investigate whether distinct questions sharing a topic entity produce
different relation preferences, SRN offers a **gold-blind** first-hop test.
It compares softmax distributions on exactly the same outgoing KG actions,
recording total variation, the maximum probability difference, and whether
the top-ranked action changes. No answer labels are inspected.

After training, with the saved model directory containing both the checkpoint
and relation NPZ:

```bash
python theseus_experiment.py --dataset kinship --hops 3 --mode diagnose \
  --glove datasets/embeddings/glove.6B.50d.txt \
  --output saved_models/theseus_kinship_single_3 \
  --sensitivity-pairs 10
```

This writes `question_sensitivity.json` and does not run the slow full
beam-search evaluation. If several different questions from one source yield
nearly identical action distributions, inspect question conditioning before
attributing poor performance to the KG or the TransE embeddings. The optional
`--sensitivity-pairs 10` argument can also be supplied to normal train or
evaluate commands to save the diagnostic after evaluation.

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

The model currently uses 50d GloVe inputs, frozen TransE relation
vectors, BiGRU question encoder, hop-specific SLP, relation attention,
semantic perceptron, GRU history, answer-level potential-based reward shaping,
and REINFORCE. It does **not** introduce evidence-path supervision or a new
loss. The executable path repairs issues in the archived third-party code,
including malformed attention tensor shapes, GRU history handling, use of the
policy-gradient loss, and label-dependent action pruning at evaluation.
Its beam search ranks by learned relation-policy scores, never gold answers.

Outputs go to `saved_models/theseus_<dataset>_<mode>_<hops>/` and include
`metrics.csv`, `glove_coverage.json`, model checkpoints, and `dev_predictions.jsonl` and
`test_predictions.jsonl` with explicit trajectories. `--seed` controls
NumPy, TensorFlow, and training order. Use `--limit N` **only for debugging**.
The original `experiment.py` and historical PathQuestion results are retained
as references, but the legacy environment is not guaranteed to run unchanged.

Run adapter tests using:

```bash
python -m unittest -v test_theseus_data.py test_prepare_embeddings.py test_glove_coverage.py test_srn_policy_updates.py
```
