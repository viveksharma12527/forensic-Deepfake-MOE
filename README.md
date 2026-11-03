# Sparsely-Gated Mixture of Experts for Speech Deepfake Detection

This project studies audio deepfake detection using a Light Convolutional Neural Network (LCNN) [1] and a sparse Mixture-of-Experts (MoE) extension. 
This repo provides: a baseline model, an experimental MoE architecture, and a common training and evaluation pipeline.

The central idea is to replace individual LCNN layers with several alternative layers, called _experts_. 
A learned _router_ selects which experts process each recording. 
This creates a model with more capacity while activating only a subset of its experts for each input. 
Whether that capacity improves detection, generalization to other datasets, or computational efficiency is something to establish experimentally.

> [1] X. Wu, R. He, Z. Sun, and T. Tan, “A light cnn for deep face representation with noisy labels,” IEEE Transactions on Information Forensics and Security, vol. 13, no. 11, pp. 2884–2896, 2018.

## The pipeline

Both models (```LCNN``` and ```LCNNMoE```) solve the same binary classification task:

| Label | Meaning                  |
| --- |--------------------------|
| `0` | Real speech (_bona fide_) |
| `1` | Fake speech     |

The pipeline is:

```text
Audio recording → mono, 16 kHz → three-second window → log-mel spectrogram → neural network → real/fake logits
```

The dataset classes load and resample audio. Short recordings are repeated to fill the window. 
Training uses random crops and samples real and fake recordings with replacement to obtain approximate class balance. 
Validation and evaluation use the middle window of each recording. 
A reported result therefore describes detection from one three-second segment per recording.

### Audio Frontend
Both models use `AudioFrontend` from [models/audio_frontend.py](models/audio_frontend.py)
to convert waveform batches into features during their forward pass. 
Each model owns an instance as `self.audio_frontend`.

The frontend applies pre-emphasis (first-order FIR high-pass filter), 
computes an 80-bin mel power spectrogram, 
applies logarithmic compression, 
and subtracts each mel bin's mean over time. 

The frontend expects mono audio sampled at 16 kHz and uses a 512-point FFT, 
a 400-sample Hamming window (25 ms), a 160-sample hop (10 ms), and a frequency range of 20 Hz – 7.6 kHz.

Input waveforms have shape `[batch, samples]`; output features have shape `[batch, 80, frames]`. 
The forward pass adds a channel dimension before passing these features to the convolutional layers. 
Feature extraction runs under `torch.no_grad()` in both models.

### Baseline and MoE model

The LCNN baseline combines convolutions, max-pooling, batch normalization, and max-feature-map (`mfm`) activations. 
A max-feature-map activation splits channels into two groups and keeps their elementwise maximum.
After adaptive pooling, fully connected layers produce the classification logits.

The MoE model follows the same structure, 
replacing nine convolutional layers and the first fully connected layer with expert banks. 
Optionally, the final classifier can also be converted to a MoE by modifying the model. 
By default, each convolutional bank contains eight experts and the fully connected bank contains four. 
Experimenting with different numbers of experts could help identify the optimal configuration.
The final classifier is shared.

Routing happens separately at each MoE layer. A router selects the top `k` experts for each recording, and the layer sums their weighted outputs. Experts are learned jointly; they are not assigned to particular datasets or synthesis methods in advance.

Both models train with cross-entropy and label smoothing of `0.2`. MoE training adds two balancing terms:

```text
loss = cross_entropy
     + lambda_importance × importance_loss
     + lambda_load × load_loss
```

* The **importance loss** term encourages a more even distribution of gate weights across experts.
* The **load loss** term encourages a more even distribution of expert usage. 

Their weights default to `1.0`. Their effect on specialization and detection performance is an experimental choice.

## Finding your way around the code

| File or directory | What to look for                                                              |
| --- |-------------------------------------------------------------------------------|
| [train_lcnn.py](train_lcnn.py) | Baseline initialization, training loop, validation, and checkpoint selection. |
| [train_lcnn_moe.py](train_lcnn_moe.py) | MoE initialization, routing settings, and auxiliary losses.                   |
| [models/lcnn.py](models/lcnn.py) | Baseline architecture.                                                        |
| [models/lcnn_moe.py](models/lcnn_moe.py) | MoE architecture.                                                             |
| [models/audio_frontend.py](models/audio_frontend.py) | Shared waveform-to-log-mel feature extraction, including pre-emphasis and temporal mean subtraction. |
| [models/moe_layer.py](models/moe_layer.py) | Dispatch of samples to selected experts and combination of their outputs.     |
| [models/gating_networks.py](models/gating_networks.py) | Router architectures, noise, top-k expert selection, and balancing losses.    |
| [src/training_utils.py](src/training_utils.py) | Dataset protocols, train/validation/test splits, sampling, and cropping.      |
| [src/schedulers.py](src/schedulers.py) | Training-time choices of `k` and routing temperature.                         |
| [config/lcnn_config.yaml](config/lcnn_config.yaml) | Experiment settings and dataset paths.                                        |
| [eval_models.py](eval_models.py) | Checkpoint loading and score generation.                                      |
| [analyze_results.py](analyze_results.py) | Conversion of scores into detection metrics.                                  |

## Running your first experiment

### 1. Prepare the environment

With Conda installed, run the following from the repository root to create and activate an environment, then install the packages used by the pipeline:

```bash
conda create --name moe-deepfake python=3.11 pip -y
conda activate moe-deepfake
python -m pip install torch torchaudio
python -m pip install numpy pandas scipy scikit-learn librosa soundfile PyYAML tqdm matplotlib
```

PyTorch and Torchaudio must be compatible with each other and with the machine's CUDA environment if you use a GPU. The repository does not currently pin dependency versions; record the working environment you use for your experiments.

Run the following commands from the repository root, since configuration and output paths are resolved relative to that directory.

### 2. Configure the data

Metadata is distributed as `meta.zip` and must be extracted before training or evaluation.
Run the following command from the repository root:

```bash
unzip meta.zip
```

Extraction creates the `meta/` directory in the repository root.

-------------

Edit the config file ([config/lcnn_config.yaml](config/lcnn_config.yaml)) to point to your dataset locations. 
Datasets are loaded from disk and must already be available. 
The default paths should work on any machine in the ISPL cluster.

The default training experiment combines **ASVspoof 2019 LA**, **Fake-or-Real**, **ADD 2022**, and **In-the-Wild**. 
Both training scripts read the selection from the config file and include its names in the checkpoint filename:

```yaml
training_datasets: [asvspoof19, fakeorreal, add2022, inthewild]
test_datasets: [asvspoof19, fakeorreal, inthewild, purdue, timit, add2022, asvspoof5, mlaad]
```

Edit these lists in the config file to select datasets independently for training and evaluation.

As of now, training supports `asvspoof19`, `fakeorreal`, `add2022`, and `inthewild`. 
Evaluation supports all eight dataset identifiers listed above.
Dataset identifiers are lowercase and case-sensitive and must match the examples above. 

Each list must be non-empty, contain only strings, contain no duplicates, and use only supported names. Structural errors are rejected before dataset loading. 
Unsupported names are detected during data preparation and may raise an error after other selected datasets have loaded.

-------------

The training loaders expect:

| Dataset | Expected dataset structure (within the root folder)                                                                                                                                                                                                                |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ASVspoof 2019 LA | Audio files in `LA/ASVspoof2019_LA_{train,dev,eval}/flac/`. Corresponding protocols in `LA/ASVspoof2019_LA_cm_protocols/`. Protocol filenames are `ASVspoof2019.LA.cm.train.trn.txt`, `ASVspoof2019.LA.cm.dev.trl.txt`, and `ASVspoof2019.LA.cm.eval.trl.txt`. |
| Fake-or-Real | `for-original-wav/{training,validation,testing}/{real,fake}/*.wav`.                                                                                                                                                                                            |
| ADD 2022 | `ADD_train_dev/{train,dev}/` and labels in `ADD_train_dev/label/{train_label,dev_label}.txt`. Test audio in `ADD2022_test/track1test/` with `ADD2022_test/track1_label.txt`.                                                                                   |
| In-the-Wild | `meta.csv` containing `file` and `label` columns, with recordings in `wavs/`.                                                                                                                                                                                  |

In-the-Wild uses a label-stratified, recording-level 60/20/20 train/validation/test split based on the configured seed. It does not enforce separation by speaker. The other training datasets use their supplied splits. The training data preparation function also reads the selected datasets' evaluation metadata, even though training itself uses only training and validation examples.

### 3. Training

```bash
python train_lcnn.py --device cuda:0
python train_lcnn_moe.py --device cuda:0
```

The only training CLI option is `-d` / `--device`. It defaults to `cpu`. 
Requesting CUDA falls back to CPU when CUDA is unavailable. 
Use `--device cpu` for a CPU run, though full training can be slow.

The config file defaults are 150 epochs, batch size 64, three-second windows, and AdamW with learning rate and weight decay both set to `0.0001`.
A cosine learning-rate scheduler decays toward `0.00001`. If you change the run length, keep `T_max` aligned with `num_epochs` for one cosine decay over the run. 
Early stopping is disabled by the default `.inf` patience.

Set `num_thread` and `prefetch_factor` to suit the machine. Their defaults, 24 workers and six prefetched batches per worker, can be demanding.
Keep `num_thread` greater than zero because the current DataLoaders use persistent workers.

Each script saves the model with the lowest validation classification loss to `save_model_folder`, defaulting to `checkpoints/LCNN_model/`. 
Filenames include the model and lowercase training dataset names, for example `LCNN_asvspoof19+fakeorreal+add2022+inthewild.pth`. 
Repeating an experiment with the same name can overwrite its checkpoint, so use separate output folders for runs you want to keep.

The baseline can initialize from `model_pretrained` within that folder if a compatible LCNN checkpoint exists. 
For a from-scratch comparison, ensure it is not unintentionally loading old weights. 

The MoE model currently trains from scratch; transferring baseline weights into experts is not implemented.

### 4. Evaluate checkpoints

```bash
python eval_models.py
```

Evaluation scans every top-level `.pth` file in `save_model_folder`, loads the model type and architecture stored in each checkpoint, and uses CUDA if available.

By default, evaluation prepares all eight datasets below. Select a subset with `test_datasets` in the config file.

| Name         | Sources                                                                                                       |
|--------------|----------------------------------------------------------------------------------------------------------------------------|
| `asvspoof19` | ASVspoof 2019 LA evaluation split: `asvspoof2019_path`.                                                                    |
| `fakeorreal` | Fake-or-Real testing split: `for_path`.                                                                                    |
| `inthewild`  | The held-out 20% split, recreated with the same seed: `InTheWild_path`.                                                    |
| `add2022`    | ADD 2022 track 1 test set: `add2022_path`.                                                                                 |
| `purdue`     | Purdue fake recordings, with LJSpeech and LibriSpeech real recordings: `purdue_path`, `ljspeech_path`, `librispeech_path`. |
| `timit`      | TIMIT-TTS fake recordings and VidTIMIT real recordings: `timit_tts_path`, `vidtimit_path`.                                 |
| `mlaad`      | English MLAAD fake recordings and sampled M-AILABS real recordings: `MLAAD_path`, `MAILABS_path`.                          |
| `asvspoof5`  | Filtered ASVspoof 5 evaluation data: `asvspoof5_meta`, `asvspoof5_path`.                                                   |

The exact file patterns and subset construction are in `prefetch_test_data`. 
* Purdue sampling targets 11,126 LibriSpeech recordings in addition to LJSpeech.
* MLAAD evaluation samples one real recording per fake recording. 
* ASVspoof 5 excludes codec-compressed data, retains a specified set of attack groups, and samples 7,760 recordings per group. Metadata is parsed with a space separator (despite its `.tsv` filename).

ASVspoof 2021 DF evaluation is disabled in the code because of reported corrupted files.
Setting its paths does not enable it.

Scores are written to `results/<dataset_name>/<checkpoint_stem>.txt` by default.
**Existing score files are replaced.**

Each line has this format, without a header:

```text
/path/to/recording.wav;0.93;0
```

The columns are **`path;probability_real;label`**. Higher scores indicate real speech, not fake speech.

### 5. Inspect the results

```bash
python analyze_results.py --root results

# Restrict the comparison to one model family
python analyze_results.py --root results --pattern 'LCNN_*.txt'
python analyze_results.py --root results --pattern 'MoE-LCNN_*.txt'
```
`--root` can be repeated for multiple result directories.

`--pattern` selects score filenames. 

The script prints per-dataset metrics as percentages. It converts scores to fake probabilities using `1 - probability_real`:

- **ROC AUC** measures discrimination across decision thresholds; higher is better.
- **Equal error rate (EER)** is the point at which false-positive and false-negative rates are equal; lower is better.
- **Balanced accuracy** averages recall for the two classes, using a fake-probability threshold of `0.5`; higher is better. Exact ties are classified as real.

Both _bona fide_ and _fake_ classes must be present for ROC/EER analysis. 

The current script prints metrics without producing plots.

## MoE implementation notes

**Training and inference may use different numbers of experts.** 

The functions in [src/schedulers.py](src/schedulers.py) return `k=2` and temperature `1.0` at every training epoch. 
**These schedulers are placeholders and should be updated in the final implementation.**

Validation and evaluation default to temperature `1.0`, through `moe.val` and `moe.eval` in the config file. 
Changing the training temperature bounds alone does not change the scheduler's output.

---------

**Routing normalization affects learning:** 
* The default `topk_softmax`  routing method applies softmax only to the selected experts. 
At `k=1`, its single gate weight is always one, so the classification loss cannot teach the router through that weight. 
Noisy load balancing can still give router gradients, which can make this easy to miss. 
Keep `k > 1` for training with this routing mode.
* The alternative `full_softmax` routing method normalizes over all experts before selection and preserves a gate-weight gradient path at `k=1`, 
but changes output scaling because selected weights are not renormalized.

See the docstrings of [`topk_softmax_routing`](models/gating_networks.py#L78) and [`full_softmax_routing`](models/gating_networks.py#L98) for more detailed explanations.
Router noise is active during training and disabled during validation and evaluation.

---------

Checkpoints contain `model_type`, `model_config`, and `state_dict`, but no optimizer state, scheduler state, or complete experiment configuration. 
They support evaluation and weight initialization, not full training resumption. 
Evaluation takes MoE `k` and temperature from the config file.

## Experimental reproducibility
The training and evaluation scripts seed Python, NumPy, and PyTorch using `seed` from the config file. GPU operations can still be nondeterministic, so runs with the same seed may produce different results.

To enable deterministic cuDNN behavior, uncomment the two lines in `seed_everything` in [`src/utils.py`](src/utils.py):

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This may reduce performance and does not guarantee determinism for every GPU operation. Stricter reproducibility may also require `torch.use_deterministic_algorithms(True)` and additional CUDA configuration; see [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

For each experiment, preserve the config file alongside the checkpoint and evaluation scores. 
Record any changes to dataset splits, sampling, or preprocessing.

For a fair comparison between the baseline and MoE models, use the same data and training settings, and report the routing settings used during training, validation, and evaluation. 
Evaluate detection performance alongside parameter count, inference time, and memory usage to assess the tradeoff between model capacity and computational cost.
