# %% [markdown]
# # VERSION TRAIN Basile Marie

# %% [markdown] {"papermill":{"duration":0.065343,"end_time":"2022-03-08T03:18:11.885586","exception":false,"start_time":"2022-03-08T03:18:11.820243","status":"completed"},"tags":[]}
# # Import Libraries

# %% [code] {"papermill":{"duration":2.632068,"end_time":"2022-03-08T03:18:14.585094","exception":false,"start_time":"2022-03-08T03:18:11.953026","status":"completed"},"tags":[],"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:31.777441Z","iopub.execute_input":"2024-06-07T08:19:31.777749Z","iopub.status.idle":"2024-06-07T08:19:50.445021Z","shell.execute_reply.started":"2024-06-07T08:19:31.777723Z","shell.execute_reply":"2024-06-07T08:19:50.444066Z"}}
import os
os.environ["KERAS_BACKEND"] = "jax"  # "jax" or "tensorflow" or "torch" 

import keras_cv
import keras
import keras.backend as K
import tensorflow as tf
import tensorflow_io as tfio

import numpy as np 
import pandas as pd

from glob import glob
from tqdm import tqdm

import librosa
import IPython.display as ipd
import librosa.display as lid

import matplotlib.pyplot as plt
import matplotlib as mpl

cmap = mpl.cm.get_cmap('coolwarm')

# %% [markdown] {"papermill":{"duration":0.065649,"end_time":"2022-03-08T03:18:14.717311","exception":false,"start_time":"2022-03-08T03:18:14.651662","status":"completed"},"tags":[]}
# ## Library Version

# %% [code] {"papermill":{"duration":0.155095,"end_time":"2022-03-08T03:18:14.939054","exception":false,"start_time":"2022-03-08T03:18:14.783959","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.450224Z","iopub.execute_input":"2024-06-07T08:19:50.450566Z","iopub.status.idle":"2024-06-07T08:19:50.455867Z","shell.execute_reply.started":"2024-06-07T08:19:50.450535Z","shell.execute_reply":"2024-06-07T08:19:50.454800Z"}}
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)

# %% [markdown] {"papermill":{"duration":0.066353,"end_time":"2022-03-08T03:18:18.099835","exception":false,"start_time":"2022-03-08T03:18:18.033482","status":"completed"},"tags":[]}
# # Configuration

# %% [code] {"papermill":{"duration":0.156464,"end_time":"2022-03-08T03:18:18.322809","exception":false,"start_time":"2022-03-08T03:18:18.166345","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.458644Z","iopub.execute_input":"2024-06-07T08:19:50.459044Z","iopub.status.idle":"2024-06-07T08:19:50.504617Z","shell.execute_reply.started":"2024-06-07T08:19:50.459004Z","shell.execute_reply":"2024-06-07T08:19:50.503915Z"}}
class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    batch_size = 64
    
    # Audio duration, sample rate, and length
    duration = 15 # second
    sample_rate = 22050
    audio_len = duration*sample_rate
    
    # STFT parameters
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    
    # Number of epochs, model name
    epochs = 10
    preset = 'efficientnetv2_b2_imagenet'
    
    # Data augmentation parameters
    augment=True

    # Class Labels for BirdCLEF 24
    class_names = sorted(os.listdir('/kaggle/input/birdclef-2024/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

# %% [markdown] {"papermill":{"duration":0.070351,"end_time":"2022-03-08T03:18:18.46058","exception":false,"start_time":"2022-03-08T03:18:18.390229","status":"completed"},"tags":[]}
# # Reproducibility

# %% [code] {"papermill":{"duration":0.153451,"end_time":"2022-03-08T03:18:18.685056","exception":false,"start_time":"2022-03-08T03:18:18.531605","status":"completed"},"tags":[],"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.505485Z","iopub.execute_input":"2024-06-07T08:19:50.505713Z","iopub.status.idle":"2024-06-07T08:19:50.510070Z","shell.execute_reply.started":"2024-06-07T08:19:50.505692Z","shell.execute_reply":"2024-06-07T08:19:50.509117Z"}}
tf.keras.utils.set_random_seed(CFG.seed)

# %% [markdown]
# # Dataset Path

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.511149Z","iopub.execute_input":"2024-06-07T08:19:50.511432Z","iopub.status.idle":"2024-06-07T08:19:50.521551Z","shell.execute_reply.started":"2024-06-07T08:19:50.511410Z","shell.execute_reply":"2024-06-07T08:19:50.520682Z"}}
BASE_PATH = '/kaggle/input/birdclef-2024'

# %% [markdown] {"papermill":{"duration":0.067107,"end_time":"2022-03-08T03:18:26.962626","exception":false,"start_time":"2022-03-08T03:18:26.895519","status":"completed"},"tags":[]}
# # Meta Data 📖

# %% [code] {"papermill":{"duration":0.241649,"end_time":"2022-03-08T03:18:27.408813","exception":false,"start_time":"2022-03-08T03:18:27.167164","status":"completed"},"tags":[],"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.522535Z","iopub.execute_input":"2024-06-07T08:19:50.522821Z","iopub.status.idle":"2024-06-07T08:19:50.798617Z","shell.execute_reply.started":"2024-06-07T08:19:50.522798Z","shell.execute_reply":"2024-06-07T08:19:50.797645Z"}}
df = pd.read_csv(f'{BASE_PATH}/train_metadata.csv')
df['filepath'] = BASE_PATH + '/train_audio/' + df.filename
df['target'] = df.primary_label.map(CFG.name2label)
df['filename'] = df.filepath.map(lambda x: x.split('/')[-1])
df['xc_id'] = df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

# Display rwos
df.head(2)

# %% [markdown]
# # EDA

# %% [markdown]
# ## Utility
# Import de l'audio
# Conversion d'un signal audio en un spectrogramme en utilisant la transformation de Fourier et l'échelle de Mel, puis normalise le spectrogramme.

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.799839Z","iopub.execute_input":"2024-06-07T08:19:50.800122Z","iopub.status.idle":"2024-06-07T08:19:50.811446Z","shell.execute_reply.started":"2024-06-07T08:19:50.800099Z","shell.execute_reply":"2024-06-07T08:19:50.810430Z"}}
def load_audio(filepath):
    audio, sr = librosa.load(filepath)
    return audio, sr

def get_spectrogram(audio):
    spec = librosa.feature.melspectrogram(y=audio, 
                                   sr=CFG.sample_rate, 
                                   n_mels=256,
                                   n_fft=2048,
                                   hop_length=512,
                                   fmax=CFG.fmax,
                                   fmin=CFG.fmin,
                                   )
    spec = librosa.power_to_db(spec, ref=1.0)
    min_ = spec.min()
    max_ = spec.max()
    if max_ != min_:
        spec = (spec - min_)/(max_ - min_)
    return spec

def display_audio(row):
    # Caption for viz
    caption = f'Id: {row.filename} | Nom Scientifique: {row.common_name} | '
    # Read audio file
    audio, sr = load_audio(row.filepath)
    # Keep fixed length audio
    audio = audio[:CFG.audio_len]
    # Spectrogram from audio
    spec = get_spectrogram(audio)
    # Display audio
    print("# Audio:")
    display(ipd.Audio(audio, rate=CFG.sample_rate))
    print('# Visualization:')
    fig, ax = plt.subplots(2, 1, figsize=(12, 2*3), sharex=True, tight_layout=True)
    fig.suptitle(caption)
    # Waveplot
    lid.waveshow(audio,
                 sr=CFG.sample_rate,
                 ax=ax[0],
                 color= cmap(0.1))
    # Specplot
    lid.specshow(spec, 
                 sr = CFG.sample_rate, 
                 hop_length=512,
                 n_fft=2048,
                 fmin=CFG.fmin,
                 fmax=CFG.fmax,
                 x_axis = 'time', 
                 y_axis = 'mel',
                 cmap = 'coolwarm',
                 ax=ax[1])
    ax[0].set_xlabel('');
    fig.show()

# %% [markdown]
# ## Sample 1

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:19:50.812440Z","iopub.execute_input":"2024-06-07T08:19:50.812711Z","iopub.status.idle":"2024-06-07T08:20:07.154946Z","shell.execute_reply.started":"2024-06-07T08:19:50.812687Z","shell.execute_reply":"2024-06-07T08:20:07.153996Z"}}
row = df.iloc[48]

# Display audio
display_audio(row)

# %% [markdown]
# ## Sample 2

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:07.158103Z","iopub.execute_input":"2024-06-07T08:20:07.158715Z","iopub.status.idle":"2024-06-07T08:20:08.868997Z","shell.execute_reply.started":"2024-06-07T08:20:07.158686Z","shell.execute_reply":"2024-06-07T08:20:08.868093Z"}}
row = df.iloc[180]

# Display audio
display_audio(row)

# %% [markdown]
# ## Sample 3

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:08.870341Z","iopub.execute_input":"2024-06-07T08:20:08.870691Z","iopub.status.idle":"2024-06-07T08:20:10.580459Z","shell.execute_reply.started":"2024-06-07T08:20:08.870660Z","shell.execute_reply":"2024-06-07T08:20:10.579506Z"}}
row = df.iloc[50]

# Display audio
display_audio(row)

# %% [markdown] {"papermill":{"duration":0.09524,"end_time":"2022-03-08T03:18:34.861029","exception":false,"start_time":"2022-03-08T03:18:34.765789","status":"completed"},"tags":[]}
# # Data Split
# Divise les données en ensembles d'entraînement et de validation en utilisant une fraction de 20% pour la validation.(surapprentissage).

# %% [code] {"papermill":{"duration":0.386301,"end_time":"2022-03-08T03:18:35.325064","exception":false,"start_time":"2022-03-08T03:18:34.938763","status":"completed"},"tags":[],"_kg_hide-input":false,"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:10.581707Z","iopub.execute_input":"2024-06-07T08:20:10.582049Z","iopub.status.idle":"2024-06-07T08:20:10.723901Z","shell.execute_reply.started":"2024-06-07T08:20:10.582018Z","shell.execute_reply":"2024-06-07T08:20:10.722942Z"}}
# Import required packages
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(df, test_size=0.2)

print(f"Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# %% [markdown] {"papermill":{"duration":0.152812,"end_time":"2022-03-08T03:18:48.676686","exception":false,"start_time":"2022-03-08T03:18:48.523874","status":"completed"},"tags":[]}
# # Data Loader

# %% [markdown]
# ## Decoders
# Ces fonctions permettent de décoder les fichiers audio, d'appliquer des transformations pour obtenir un spectrogramme, et de préparer les données pour l'entraînement.

# %% [code] {"papermill":{"duration":0.251237,"end_time":"2022-03-08T03:18:49.079346","exception":false,"start_time":"2022-03-08T03:18:48.828109","status":"completed"},"tags":[],"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:10.725320Z","iopub.execute_input":"2024-06-07T08:20:10.725674Z","iopub.status.idle":"2024-06-07T08:20:10.744220Z","shell.execute_reply.started":"2024-06-07T08:20:10.725641Z","shell.execute_reply":"2024-06-07T08:20:10.743363Z"}}
# Decodes Audio
def build_decoder(with_labels=True, dim=1024):
    def get_audio(filepath):
        file_bytes = tf.io.read_file(filepath)
        audio = tfio.audio.decode_vorbis(file_bytes)  # decode .ogg file
        audio = tf.cast(audio, tf.float32)
        if tf.shape(audio)[1] > 1:  # stereo -> mono
            audio = audio[..., 0:1]
        audio = tf.squeeze(audio, axis=-1)
        return audio

    def crop_or_pad(audio, target_len, pad_mode="constant"):
        audio_len = tf.shape(audio)[0]
        diff_len = abs(
            target_len - audio_len
        )  # find difference between target and audio length
        if audio_len < target_len:  # do padding if audio length is shorter
            pad1 = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            pad2 = diff_len - pad1
            audio = tf.pad(audio, paddings=[[pad1, pad2]], mode=pad_mode)
        elif audio_len > target_len:  # do cropping if audio length is larger
            idx = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            audio = audio[idx : (idx + target_len)]
        return tf.reshape(audio, [target_len])

    def apply_preproc(spec):
        # Standardize
        mean = tf.math.reduce_mean(spec)
        std = tf.math.reduce_std(spec)
        spec = tf.where(tf.math.equal(std, 0), spec - mean, (spec - mean) / std)

        # Normalize using Min-Max
        min_val = tf.math.reduce_min(spec)
        max_val = tf.math.reduce_max(spec)
        spec = tf.where(
            tf.math.equal(max_val - min_val, 0),
            spec - min_val,
            (spec - min_val) / (max_val - min_val),
        )
        return spec

    def get_target(target):
        target = tf.reshape(target, [1])
        target = tf.cast(tf.one_hot(target, CFG.num_classes), tf.float32)
        target = tf.reshape(target, [CFG.num_classes])
        return target

    def decode(path):
        # Load audio file
        audio = get_audio(path)
        # Crop or pad audio to keep a fixed length
        audio = crop_or_pad(audio, dim)
        # Audio to Spectrogram
        spec = keras.layers.MelSpectrogram(
            num_mel_bins=CFG.img_size[0],
            fft_length=CFG.nfft,
            sequence_stride=CFG.hop_length,
            sampling_rate=CFG.sample_rate,
        )(audio)
        # Apply normalization and standardization
        spec = apply_preproc(spec)
        # Spectrogram to 3 channel image (for imagenet)
        spec = tf.tile(spec[..., None], [1, 1, 3])
        spec = tf.reshape(spec, [*CFG.img_size, 3])
        return spec

    def decode_with_labels(path, label):
        label = get_target(label)
        return decode(path), label

    return decode_with_labels if with_labels else decode

# %% [markdown] {"papermill":{"duration":0.150182,"end_time":"2022-03-08T03:18:49.38214","exception":false,"start_time":"2022-03-08T03:18:49.231958","status":"completed"},"tags":[]}
# ## Augmenters
# 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:10.745607Z","iopub.execute_input":"2024-06-07T08:20:10.745952Z","iopub.status.idle":"2024-06-07T08:20:10.757172Z","shell.execute_reply.started":"2024-06-07T08:20:10.745921Z","shell.execute_reply":"2024-06-07T08:20:10.756253Z"}}
def build_augmenter():
    augmenters = [
        keras_cv.layers.MixUp(alpha=0.4),
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                     width_factor=(0.06, 0.12)), # time-masking
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                     width_factor=(1.0, 1.0)), # freq-masking
    ]
    
    def augment(img, label):
        data = {"images":img, "labels":label}
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.35:
                data = augmenter(data, training=True)
        return data["images"], data["labels"]
    
    return augment

# %% [markdown] {"papermill":{"duration":0.152217,"end_time":"2022-03-08T03:18:50.097623","exception":false,"start_time":"2022-03-08T03:18:49.945406","status":"completed"},"tags":[]}
# ## Data Pipeline
# 

# %% [code] {"papermill":{"duration":0.240881,"end_time":"2022-03-08T03:18:50.489717","exception":false,"start_time":"2022-03-08T03:18:50.248836","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:10.758318Z","iopub.execute_input":"2024-06-07T08:20:10.758636Z","iopub.status.idle":"2024-06-07T08:20:10.767575Z","shell.execute_reply.started":"2024-06-07T08:20:10.758601Z","shell.execute_reply":"2024-06-07T08:20:10.766691Z"}}
def build_dataset(paths, labels=None, batch_size=32, 
                  decode_fn=None, augment_fn=None, cache=True,
                  augment=False, shuffle=2048):

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None, dim=CFG.audio_len)

    if augment_fn is None:
        augment_fn = build_augmenter()
        
    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths,) if labels is None else (paths, labels)
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache() if cache else ds
    if shuffle:
        opt = tf.data.Options()
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds

# %% [markdown]
# ## Build Train and Valid Dataloaders

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:10.768700Z","iopub.execute_input":"2024-06-07T08:20:10.768954Z","iopub.status.idle":"2024-06-07T08:20:15.127263Z","shell.execute_reply.started":"2024-06-07T08:20:10.768932Z","shell.execute_reply":"2024-06-07T08:20:15.126489Z"}}
# Train
train_paths = train_df.filepath.values
train_labels = train_df.target.values
train_ds = build_dataset(train_paths, train_labels, batch_size=CFG.batch_size,
                         shuffle=True, augment=CFG.augment)

# Valid
valid_paths = valid_df.filepath.values
valid_labels = valid_df.target.values
valid_ds = build_dataset(valid_paths, valid_labels, batch_size=CFG.batch_size,
                         shuffle=False, augment=False)

# %% [markdown]
# # Visualization 
# 

# %% [code] {"papermill":{"duration":0.328513,"end_time":"2022-03-08T03:19:59.512224","exception":false,"start_time":"2022-03-08T03:19:59.183711","status":"completed"},"tags":[],"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:15.128787Z","iopub.execute_input":"2024-06-07T08:20:15.129122Z","iopub.status.idle":"2024-06-07T08:20:15.136726Z","shell.execute_reply.started":"2024-06-07T08:20:15.129088Z","shell.execute_reply":"2024-06-07T08:20:15.135813Z"}}
def plot_batch(batch, row=3, col=3, label2name=None,):
    """Plot one batch data"""
    if isinstance(batch, tuple) or isinstance(batch, list):
        specs, tars = batch
    else:
        specs = batch
        tars = None
    plt.figure(figsize=(col*5, row*3))
    for idx in range(row*col):
        ax = plt.subplot(row, col, idx+1)
        lid.specshow(np.array(specs[idx, ..., 0]), 
                     n_fft=CFG.nfft, 
                     hop_length=CFG.hop_length, 
                     sr=CFG.sample_rate,
                     x_axis='time',
                     y_axis='mel',
                     cmap='coolwarm')
        if tars is not None:
            label = tars[idx].numpy().argmax()
            name = label2name[label]
            plt.title(name)
    plt.tight_layout()
    plt.show()

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:15.138021Z","iopub.execute_input":"2024-06-07T08:20:15.138307Z","iopub.status.idle":"2024-06-07T08:20:29.399759Z","shell.execute_reply.started":"2024-06-07T08:20:15.138267Z","shell.execute_reply":"2024-06-07T08:20:29.398798Z"}}
sample_ds = train_ds.take(100)
batch = next(iter(sample_ds))
plot_batch(batch, label2name=CFG.label2name)

# %% [markdown] {"papermill":{"duration":0.182769,"end_time":"2022-03-08T03:20:04.861966","exception":false,"start_time":"2022-03-08T03:20:04.679197","status":"completed"},"tags":[]}
# # Modeling
# 

# %% [code] {"papermill":{"duration":1.239321,"end_time":"2022-03-08T03:20:06.281118","exception":false,"start_time":"2022-03-08T03:20:05.041797","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-07T08:20:29.400900Z","iopub.execute_input":"2024-06-07T08:20:29.401181Z","iopub.status.idle":"2024-06-07T08:20:47.581926Z","shell.execute_reply.started":"2024-06-07T08:20:29.401156Z","shell.execute_reply":"2024-06-07T08:20:47.580974Z"},"jupyter":{"outputs_hidden":true}}
# Create an input layer for the model
inp = keras.layers.Input(shape=(None, None, 3))
# Pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    CFG.preset,
)
out = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=CFG.num_classes,
    name="classifier"
)(inp)
# Build model
model = keras.models.Model(inputs=inp, outputs=out)
# Compile model with optimizer, loss and metrics
model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
              metrics=[keras.metrics.AUC(name='auc')],
             )
model.summary()

# %% [markdown]
# # LR Schedule

# %% [code] {"_kg_hide-input":true,"papermill":{"duration":0.510014,"end_time":"2022-03-08T03:20:45.290695","exception":false,"start_time":"2022-03-08T03:20:44.780681","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-07T08:20:47.583207Z","iopub.execute_input":"2024-06-07T08:20:47.584172Z","iopub.status.idle":"2024-06-07T08:20:47.596517Z","shell.execute_reply.started":"2024-06-07T08:20:47.584138Z","shell.execute_reply":"2024-06-07T08:20:47.595518Z"}}
import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

# %% [code] {"execution":{"iopub.status.busy":"2024-06-07T08:20:47.598034Z","iopub.execute_input":"2024-06-07T08:20:47.598404Z","iopub.status.idle":"2024-06-07T08:20:47.881208Z","shell.execute_reply.started":"2024-06-07T08:20:47.598374Z","shell.execute_reply":"2024-06-07T08:20:47.880093Z"}}
lr_cb = get_lr_callback(CFG.batch_size, plot=True)

# %% [markdown]
# # Model Checkpoint

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:47.882676Z","iopub.execute_input":"2024-06-07T08:20:47.883029Z","iopub.status.idle":"2024-06-07T08:20:47.888912Z","shell.execute_reply.started":"2024-06-07T08:20:47.882995Z","shell.execute_reply":"2024-06-07T08:20:47.887769Z"}}
ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.weights.h5",
                                         monitor='val_auc',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')

# %% [markdown]
# # Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T08:20:47.890152Z","iopub.execute_input":"2024-06-07T08:20:47.890533Z","iopub.status.idle":"2024-06-07T09:29:22.961876Z","shell.execute_reply.started":"2024-06-07T08:20:47.890500Z","shell.execute_reply":"2024-06-07T09:29:22.960899Z"}}
history = model.fit(
    train_ds, 
    validation_data=valid_ds, 
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb], 
    verbose=1
)

# %% [markdown]
# ## Result Summary
# Prépare les DataFrames d'entraînement et de validation et les convertit en DataLoaders TensorFlow prêts pour l'entraînement et la validation du modèle.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-07T09:29:22.963505Z","iopub.execute_input":"2024-06-07T09:29:22.963797Z","iopub.status.idle":"2024-06-07T09:29:22.969549Z","shell.execute_reply.started":"2024-06-07T09:29:22.963770Z","shell.execute_reply":"2024-06-07T09:29:22.968462Z"}}
best_epoch = np.argmax(history.history["val_auc"])
best_score = history.history["val_auc"][best_epoch]
print('>>> Best AUC: ', best_score)
print('>>> Best Epoch: ', best_epoch+1)
