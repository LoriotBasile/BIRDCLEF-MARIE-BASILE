# %% [markdown]
# # VERSION TRAIN 
# 

# %% [markdown] {"papermill":{"duration":0.065343,"end_time":"2022-03-08T03:18:11.885586","exception":false,"start_time":"2022-03-08T03:18:11.820243","status":"completed"},"tags":[]}
# # Import Libraries

# %% [code] {"papermill":{"duration":2.632068,"end_time":"2022-03-08T03:18:14.585094","exception":false,"start_time":"2022-03-08T03:18:11.953026","status":"completed"},"tags":[],"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-06-06T09:52:41.561392Z","iopub.execute_input":"2024-06-06T09:52:41.561667Z","iopub.status.idle":"2024-06-06T09:53:00.230168Z","shell.execute_reply.started":"2024-06-06T09:52:41.561642Z","shell.execute_reply":"2024-06-06T09:53:00.228970Z"}}
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

# %% [code] {"papermill":{"duration":0.155095,"end_time":"2022-03-08T03:18:14.939054","exception":false,"start_time":"2022-03-08T03:18:14.783959","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-06T09:53:00.231891Z","iopub.execute_input":"2024-06-06T09:53:00.232437Z","iopub.status.idle":"2024-06-06T09:53:00.237242Z","shell.execute_reply.started":"2024-06-06T09:53:00.232410Z","shell.execute_reply":"2024-06-06T09:53:00.236418Z"}}
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)

# %% [markdown] {"papermill":{"duration":0.066353,"end_time":"2022-03-08T03:18:18.099835","exception":false,"start_time":"2022-03-08T03:18:18.033482","status":"completed"},"tags":[]}
# # Configuration 

# %% [code] {"papermill":{"duration":0.156464,"end_time":"2022-03-08T03:18:18.322809","exception":false,"start_time":"2022-03-08T03:18:18.166345","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-06T09:53:00.238316Z","iopub.execute_input":"2024-06-06T09:53:00.238580Z","iopub.status.idle":"2024-06-06T09:53:00.277417Z","shell.execute_reply.started":"2024-06-06T09:53:00.238557Z","shell.execute_reply":"2024-06-06T09:53:00.276724Z"}}
class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    batch_size = 64
    
    # Audio duration, sample rate, and length
    duration = 15 # second
    sample_rate = 32000
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
# 

# %% [code] {"papermill":{"duration":0.153451,"end_time":"2022-03-08T03:18:18.685056","exception":false,"start_time":"2022-03-08T03:18:18.531605","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:00.279585Z","iopub.execute_input":"2024-06-06T09:53:00.279892Z","iopub.status.idle":"2024-06-06T09:53:00.284443Z","shell.execute_reply.started":"2024-06-06T09:53:00.279867Z","shell.execute_reply":"2024-06-06T09:53:00.283224Z"}}
tf.keras.utils.set_random_seed(CFG.seed)

# %% [markdown]
# # Dataset Path

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:00.285304Z","iopub.execute_input":"2024-06-06T09:53:00.285539Z","iopub.status.idle":"2024-06-06T09:53:00.293586Z","shell.execute_reply.started":"2024-06-06T09:53:00.285515Z","shell.execute_reply":"2024-06-06T09:53:00.292856Z"}}
BASE_PATH = '/kaggle/input/birdclef-2024'

# %% [markdown] {"papermill":{"duration":0.067107,"end_time":"2022-03-08T03:18:26.962626","exception":false,"start_time":"2022-03-08T03:18:26.895519","status":"completed"},"tags":[]}
# # Meta Data 📖

# %% [code] {"papermill":{"duration":0.241649,"end_time":"2022-03-08T03:18:27.408813","exception":false,"start_time":"2022-03-08T03:18:27.167164","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:00.294742Z","iopub.execute_input":"2024-06-06T09:53:00.295023Z","iopub.status.idle":"2024-06-06T09:53:00.536878Z","shell.execute_reply.started":"2024-06-06T09:53:00.295000Z","shell.execute_reply":"2024-06-06T09:53:00.535861Z"}}
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

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:00.537914Z","iopub.execute_input":"2024-06-06T09:53:00.538187Z","iopub.status.idle":"2024-06-06T09:53:00.549369Z","shell.execute_reply.started":"2024-06-06T09:53:00.538164Z","shell.execute_reply":"2024-06-06T09:53:00.548331Z"}}
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

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:00.550514Z","iopub.execute_input":"2024-06-06T09:53:00.550826Z","iopub.status.idle":"2024-06-06T09:53:16.660490Z","shell.execute_reply.started":"2024-06-06T09:53:00.550780Z","shell.execute_reply":"2024-06-06T09:53:16.659679Z"}}
row = df.iloc[48]

# Display audio
display_audio(row)

# %% [markdown]
# ## Sample 2

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:58:34.803015Z","iopub.execute_input":"2024-06-06T09:58:34.803429Z","iopub.status.idle":"2024-06-06T09:58:37.667824Z","shell.execute_reply.started":"2024-06-06T09:58:34.803394Z","shell.execute_reply":"2024-06-06T09:58:37.666873Z"}}
row = df.iloc[180]

# Display audio
display_audio(row)

# %% [markdown]
# ## Sample 3

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:18.304187Z","iopub.execute_input":"2024-06-06T09:53:18.304789Z","iopub.status.idle":"2024-06-06T09:53:19.949677Z","shell.execute_reply.started":"2024-06-06T09:53:18.304756Z","shell.execute_reply":"2024-06-06T09:53:19.948683Z"}}
row = df.iloc[50]

# Display audio
display_audio(row)

# %% [markdown] {"papermill":{"duration":0.09524,"end_time":"2022-03-08T03:18:34.861029","exception":false,"start_time":"2022-03-08T03:18:34.765789","status":"completed"},"tags":[]}
# # Data Split
# Following code will split the data into folds using target stratification.
# > **Note:** Some classess have too few samples thus not each fold contains all the classes. 

# %% [code] {"papermill":{"duration":0.386301,"end_time":"2022-03-08T03:18:35.325064","exception":false,"start_time":"2022-03-08T03:18:34.938763","status":"completed"},"tags":[],"_kg_hide-input":false,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:19.950782Z","iopub.execute_input":"2024-06-06T09:53:19.951136Z","iopub.status.idle":"2024-06-06T09:53:20.096513Z","shell.execute_reply.started":"2024-06-06T09:53:19.951111Z","shell.execute_reply":"2024-06-06T09:53:20.095596Z"}}
# Import required packages
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(df, test_size=0.2)

print(f"Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# %% [markdown] {"papermill":{"duration":0.152812,"end_time":"2022-03-08T03:18:48.676686","exception":false,"start_time":"2022-03-08T03:18:48.523874","status":"completed"},"tags":[]}
# # Data Loader 

# %% [markdown]
# ## Decoders
# 
# The following code will decode the raw audio from `.ogg` file and also decode the spectrogram from the `audio` file. Additionally, we will apply Z-Score standardization and Min-Max normalization to ensure consistent inputs to the model.
# 

# %% [code] {"papermill":{"duration":0.251237,"end_time":"2022-03-08T03:18:49.079346","exception":false,"start_time":"2022-03-08T03:18:48.828109","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:20.098063Z","iopub.execute_input":"2024-06-06T09:53:20.098564Z","iopub.status.idle":"2024-06-06T09:53:20.114577Z","shell.execute_reply.started":"2024-06-06T09:53:20.098528Z","shell.execute_reply":"2024-06-06T09:53:20.113605Z"}}
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
# Following code will apply augmentations to spectrogram data. In this notebook, we will use MixUp, CutOut (TimeMasking and FreqMasking) from KerasCV.
# 
# > Note that, these augmentations will be applied to batch of spectrograms rather than single spectrograms.

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:53:20.115879Z","iopub.execute_input":"2024-06-06T09:53:20.116688Z","iopub.status.idle":"2024-06-06T09:53:20.127484Z","shell.execute_reply.started":"2024-06-06T09:53:20.116662Z","shell.execute_reply":"2024-06-06T09:53:20.126650Z"}}
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
# Following code builds the complete pipeline of the data flow. It uses `tf.data.Dataset` for data processing. Here are some cool features of `tf.data`,
# * We can build complex input pipelines from simple, reusable pieces using`tf.data` API . For example, the pipeline for an audio model might aggregate data from files in a distributed file system, apply random transformation/augmentation to each audio/spectrogram, and merge randomly selected data into a batch for training.
# * Moreover `tf.data` API provides a `tf.data.Dataset` feature that represents a sequence of components where each component comprises one or more pieces. For instance, in an audio pipeline, a component might be a single training example, with a pair of tensor pieces representing the audio and its label.
# 
# Check out this [doc](https://www.tensorflow.org/guide/data) if you want to learn more about `tf.data`.

# %% [code] {"papermill":{"duration":0.240881,"end_time":"2022-03-08T03:18:50.489717","exception":false,"start_time":"2022-03-08T03:18:50.248836","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-06T09:53:20.128541Z","iopub.execute_input":"2024-06-06T09:53:20.128840Z","iopub.status.idle":"2024-06-06T09:53:20.137707Z","shell.execute_reply.started":"2024-06-06T09:53:20.128794Z","shell.execute_reply":"2024-06-06T09:53:20.136860Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:53:20.138914Z","iopub.execute_input":"2024-06-06T09:53:20.139400Z","iopub.status.idle":"2024-06-06T09:53:24.298021Z","shell.execute_reply.started":"2024-06-06T09:53:20.139375Z","shell.execute_reply":"2024-06-06T09:53:24.296736Z"}}
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
# To ensure our pipeline is generating **spectrogram** and its associate **label** correctly, we'll check some samples from a batch.

# %% [code] {"papermill":{"duration":0.328513,"end_time":"2022-03-08T03:19:59.512224","exception":false,"start_time":"2022-03-08T03:19:59.183711","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:24.299348Z","iopub.execute_input":"2024-06-06T09:53:24.299739Z","iopub.status.idle":"2024-06-06T09:53:24.309583Z","shell.execute_reply.started":"2024-06-06T09:53:24.299704Z","shell.execute_reply":"2024-06-06T09:53:24.308651Z"}}
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

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:24.310753Z","iopub.execute_input":"2024-06-06T09:53:24.311709Z","iopub.status.idle":"2024-06-06T09:53:38.742077Z","shell.execute_reply.started":"2024-06-06T09:53:24.311675Z","shell.execute_reply":"2024-06-06T09:53:38.741111Z"}}
sample_ds = train_ds.take(100)
batch = next(iter(sample_ds))
plot_batch(batch, label2name=CFG.label2name)

# %% [markdown] {"papermill":{"duration":0.182769,"end_time":"2022-03-08T03:20:04.861966","exception":false,"start_time":"2022-03-08T03:20:04.679197","status":"completed"},"tags":[]}
# # 🤖 Modeling
# 
# Building a model for an audio recognition task with spectrograms as input is quite straightforward, as it is very similar to image classification. This is because the shape of spectrogram data is very similar to image data. In this notebook, to perform the audio recognition task, we will utilize the `EfficientNetV2` ImageNet-pretrained model as the backbone. Even though this backbone is pretrained with ImageNet data instead of spectrogram data, we can leverage transfer learning to adapt it to our spectrogram-based task.
# 
# > Note that we can train our model on any duration audio file (here we are using `10 seconds`), but we will always infer on `5-second` audio files (as per competition rules). To facilitate this, we have set the model input shape to `(None, None, 3)`, which will allow us to have variable-length input during training and inference.
# 
# 
# In case you are wondering, **Why not train and infer on both `5-second`?** In the train data, we have long audio files, but we are not sure which part of the audio contains the labeled bird's song. In other words, this is weakly labeled. To ensure the provided label is accurately suited to the audio, we are using a larger audio size than `5 seconds`. You are welcome to try out different audio lengths for training.
# 

# %% [code] {"papermill":{"duration":1.239321,"end_time":"2022-03-08T03:20:06.281118","exception":false,"start_time":"2022-03-08T03:20:05.041797","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T09:53:38.743454Z","iopub.execute_input":"2024-06-06T09:53:38.743804Z","iopub.status.idle":"2024-06-06T09:53:55.645633Z","shell.execute_reply.started":"2024-06-06T09:53:38.743775Z","shell.execute_reply":"2024-06-06T09:53:55.644705Z"}}
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
# 

# %% [code] {"_kg_hide-input":true,"papermill":{"duration":0.510014,"end_time":"2022-03-08T03:20:45.290695","exception":false,"start_time":"2022-03-08T03:20:44.780681","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-06T09:53:55.646990Z","iopub.execute_input":"2024-06-06T09:53:55.647535Z","iopub.status.idle":"2024-06-06T09:53:55.657156Z","shell.execute_reply.started":"2024-06-06T09:53:55.647508Z","shell.execute_reply":"2024-06-06T09:53:55.656288Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:53:55.658139Z","iopub.execute_input":"2024-06-06T09:53:55.658602Z","iopub.status.idle":"2024-06-06T09:53:55.914411Z","shell.execute_reply.started":"2024-06-06T09:53:55.658577Z","shell.execute_reply":"2024-06-06T09:53:55.913580Z"}}
lr_cb = get_lr_callback(CFG.batch_size, plot=True)

# %% [markdown]
# # Model Checkpoint 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:53:55.915345Z","iopub.execute_input":"2024-06-06T09:53:55.915585Z","iopub.status.idle":"2024-06-06T09:53:55.920084Z","shell.execute_reply.started":"2024-06-06T09:53:55.915563Z","shell.execute_reply":"2024-06-06T09:53:55.919125Z"}}
ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.weights.h5",
                                         monitor='val_auc',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')

# %% [markdown]
# # Training 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:53:55.921319Z","iopub.execute_input":"2024-06-06T09:53:55.922138Z","iopub.status.idle":"2024-06-06T09:58:19.338570Z","shell.execute_reply.started":"2024-06-06T09:53:55.922112Z","shell.execute_reply":"2024-06-06T09:58:19.335375Z"}}
history = model.fit(
    train_ds, 
    validation_data=valid_ds, 
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb], 
    verbose=1
)

# %% [markdown]
# ## Result Summary

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T09:58:19.341347Z","iopub.status.idle":"2024-06-06T09:58:19.343890Z","shell.execute_reply.started":"2024-06-06T09:58:19.343621Z","shell.execute_reply":"2024-06-06T09:58:19.343642Z"}}
best_epoch = np.argmax(history.history["val_auc"])
best_score = history.history["val_auc"][best_epoch]
print('>>> Best AUC: ', best_score)
print('>>> Best Epoch: ', best_epoch+1)
