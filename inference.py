# %% [markdown]
# # PROJET BIRDCLEF Marie & Basile

# %% [markdown]
# BirdCLEF 2024 avec KerasCV et Keras
# L'objectif de cette compétition est d'identifier les espèces d'oiseaux indiens peu étudiées par leurs appels.
# 
# Ce notebook, réalisé par Marie et Basile, vous guide à travers le processus d'inférence d'un modèle de Deep Learning pour reconnaître les espèces d'oiseaux par leurs chants (données audio). Étant donné que l'inférence nécessite de fonctionner uniquement sur le CPU, nous avons dû créer des notebooks séparés pour l'entraînement et l'inférence. Vous pouvez trouver le notebook d'entraînement ici. Pour rappel, le notebook d'entraînement utilise le backbone EfficientNetV2 de KerasCV sur le dataset de la compétition. Ce notebook montre également comment convertir les données audio en spectrogrammes Mel à l'aide de Keras.
# 
# <u>Fait amusant</u> : Les notebooks d'entraînement et d'inférence sont agnostiques par rapport au backend, prenant en charge TensorFlow, PyTorch et JAX. Utiliser KerasCV et Keras nous permet de choisir notre backend préféré. Explorez plus de détails sur Keras.
# 
# Dans ce notebook, vous apprendrez :
# 
# À concevoir un pipeline de données pour les données audio, y compris la conversion audio en spectrogrammes.
# À charger les données efficacement en utilisant tf.data.
# À créer le modèle en utilisant les presets de KerasCV.
# À inférer le modèle entraîné.
# Note : Pour une compréhension plus approfondie de KerasCV, référez-vous aux guides KerasCV.

# %% [markdown] {"papermill":{"duration":0.065343,"end_time":"2022-03-08T03:18:11.885586","exception":false,"start_time":"2022-03-08T03:18:11.820243","status":"completed"},"tags":[]}
# # Import Libraries 

# %% [code] {"papermill":{"duration":2.632068,"end_time":"2022-03-08T03:18:14.585094","exception":false,"start_time":"2022-03-08T03:18:11.953026","status":"completed"},"tags":[],"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-06-06T11:37:47.471736Z","iopub.execute_input":"2024-06-06T11:37:47.472110Z","iopub.status.idle":"2024-06-06T11:38:17.523324Z","shell.execute_reply.started":"2024-06-06T11:37:47.472069Z","shell.execute_reply":"2024-06-06T11:38:17.522329Z"}}
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # "jax" or "tensorflow" or "torch" 

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

# %% [markdown] {"papermill":{"duration":0.066353,"end_time":"2022-03-08T03:18:18.099835","exception":false,"start_time":"2022-03-08T03:18:18.033482","status":"completed"},"tags":[]}
# # Configuration

# %% [code] {"papermill":{"duration":0.156464,"end_time":"2022-03-08T03:18:18.322809","exception":false,"start_time":"2022-03-08T03:18:18.166345","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-06-06T11:38:17.525257Z","iopub.execute_input":"2024-06-06T11:38:17.525772Z","iopub.status.idle":"2024-06-06T11:38:17.546224Z","shell.execute_reply.started":"2024-06-06T11:38:17.525745Z","shell.execute_reply":"2024-06-06T11:38:17.545287Z"}}
class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    
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
    preset = 'efficientnetv2_b2_imagenet'

    # Class Labels for BirdCLEF 24
    class_names = sorted(os.listdir('/kaggle/input/birdclef-2024/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

# %% [markdown] {"papermill":{"duration":0.070351,"end_time":"2022-03-08T03:18:18.46058","exception":false,"start_time":"2022-03-08T03:18:18.390229","status":"completed"},"tags":[]}
# # Reproducibility

# %% [code] {"papermill":{"duration":0.153451,"end_time":"2022-03-08T03:18:18.685056","exception":false,"start_time":"2022-03-08T03:18:18.531605","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T11:38:17.547292Z","iopub.execute_input":"2024-06-06T11:38:17.547589Z","iopub.status.idle":"2024-06-06T11:38:17.554194Z","shell.execute_reply.started":"2024-06-06T11:38:17.547556Z","shell.execute_reply":"2024-06-06T11:38:17.553385Z"}}
tf.keras.utils.set_random_seed(CFG.seed)

# %% [markdown]
# # Dataset Path

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T11:38:17.555286Z","iopub.execute_input":"2024-06-06T11:38:17.555872Z","iopub.status.idle":"2024-06-06T11:38:17.567153Z","shell.execute_reply.started":"2024-06-06T11:38:17.555840Z","shell.execute_reply":"2024-06-06T11:38:17.566452Z"}}
BASE_PATH = '/kaggle/input/birdclef-2024'

# %% [markdown] {"papermill":{"duration":0.067107,"end_time":"2022-03-08T03:18:26.962626","exception":false,"start_time":"2022-03-08T03:18:26.895519","status":"completed"},"tags":[]}
# # Test Data 

# %% [code] {"papermill":{"duration":0.241649,"end_time":"2022-03-08T03:18:27.408813","exception":false,"start_time":"2022-03-08T03:18:27.167164","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T11:38:17.569714Z","iopub.execute_input":"2024-06-06T11:38:17.569964Z","iopub.status.idle":"2024-06-06T11:38:17.753641Z","shell.execute_reply.started":"2024-06-06T11:38:17.569944Z","shell.execute_reply":"2024-06-06T11:38:17.752816Z"}}
test_paths = glob(f'{BASE_PATH}/test_soundscapes/*ogg')
# During commit use `unlabeled` data as there is no `test` data.
# During submission `test` data will automatically be populated.
if len(test_paths)==0:
    test_paths = glob(f'{BASE_PATH}/unlabeled_soundscapes/*ogg')[:10]
test_df = pd.DataFrame(test_paths, columns=['filepath'])
test_df.head()

# %% [markdown] {"papermill":{"duration":0.182769,"end_time":"2022-03-08T03:20:04.861966","exception":false,"start_time":"2022-03-08T03:20:04.679197","status":"completed"},"tags":[]}
# # Modeling

# %% [code] {"papermill":{"duration":1.239321,"end_time":"2022-03-08T03:20:06.281118","exception":false,"start_time":"2022-03-08T03:20:05.041797","status":"completed"},"tags":[],"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-06-06T11:38:17.754731Z","iopub.execute_input":"2024-06-06T11:38:17.755057Z","iopub.status.idle":"2024-06-06T11:38:25.785413Z","shell.execute_reply.started":"2024-06-06T11:38:17.755012Z","shell.execute_reply":"2024-06-06T11:38:25.784425Z"}}
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
# Load weights of trained model
model.load_weights("/kaggle/input/birdclef24-kerascv-starter-train/best_model.weights.h5")

# %% [markdown]
# # Data Loader 
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T11:38:25.787125Z","iopub.execute_input":"2024-06-06T11:38:25.787792Z","iopub.status.idle":"2024-06-06T11:38:25.799782Z","shell.execute_reply.started":"2024-06-06T11:38:25.787757Z","shell.execute_reply":"2024-06-06T11:38:25.798953Z"}}
# Decodes Audio
def build_decoder(with_labels=True, dim=1024):
    def get_audio(filepath):
        file_bytes = tf.io.read_file(filepath)
        audio = tfio.audio.decode_vorbis(file_bytes) # decode .ogg file
        audio = tf.cast(audio, tf.float32)
        if tf.shape(audio)[1]>1: # stereo -> mono
            audio = audio[...,0:1]
        audio = tf.squeeze(audio, axis=-1)
        return audio
    
    def create_frames(audio, duration=5, sr=32000):
        frame_size = int(duration * sr)
        audio = tf.pad(audio[..., None], [[0, tf.shape(audio)[0] % frame_size], [0, 0]]) # pad the end
        audio = tf.squeeze(audio) # remove extra dimension added for padding
        frames = tf.reshape(audio, [-1, frame_size]) # shape: [num_frames, frame_size]
        return frames
    
    def apply_preproc(spec):
        # Standardize
        mean = tf.math.reduce_mean(spec)
        std = tf.math.reduce_std(spec)
        spec = tf.where(tf.math.equal(std, 0), spec - mean, (spec - mean) / std)

        # Normalize using Min-Max
        min_val = tf.math.reduce_min(spec)
        max_val = tf.math.reduce_max(spec)
        spec = tf.where(tf.math.equal(max_val - min_val, 0), spec - min_val,
                              (spec - min_val) / (max_val - min_val))
        return spec

    def decode(path):
        # Load audio file
        audio = get_audio(path)
        # Split audio file into frames with each having 5 seecond duration
        audio = create_frames(audio)
        # Convert audio to spectrogram
        spec = keras.layers.MelSpectrogram(num_mel_bins=CFG.img_size[0],
                                             fft_length=CFG.nfft, 
                                              sequence_stride=CFG.hop_length, 
                                              sampling_rate=CFG.sample_rate)(audio)
        # Apply normalization and standardization
        spec = apply_preproc(spec)
        # Covnert spectrogram to 3 channel image (for imagenet)
        spec = tf.tile(spec[..., None], [1, 1, 1, 3])
        return spec
    
    return decode

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T11:38:25.801017Z","iopub.execute_input":"2024-06-06T11:38:25.801361Z","iopub.status.idle":"2024-06-06T11:38:25.815643Z","shell.execute_reply.started":"2024-06-06T11:38:25.801332Z","shell.execute_reply":"2024-06-06T11:38:25.814819Z"}}
# Build data loader
def build_dataset(paths, batch_size=1, decode_fn=None, cache=False):
    if decode_fn is None:
        decode_fn = build_decoder(dim=CFG.audio_len) # decoder
    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths,)
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO) # decode audio to spectrograms then create frames
    ds = ds.cache() if cache else ds # cache files
    ds = ds.batch(batch_size, drop_remainder=False) # create batches
    ds = ds.prefetch(AUTO)
    return ds

# %% [markdown]
# # Inference 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T11:38:25.816745Z","iopub.execute_input":"2024-06-06T11:38:25.816994Z","iopub.status.idle":"2024-06-06T11:38:59.119999Z","shell.execute_reply.started":"2024-06-06T11:38:25.816972Z","shell.execute_reply":"2024-06-06T11:38:59.119155Z"}}
# Initialize empty list to store ids
ids = []

# Initialize empty array to store predictions
preds = np.empty(shape=(0, CFG.num_classes), dtype='float32')

# Build test dataset
test_paths = test_df.filepath.tolist()
test_ds = build_dataset(paths=test_paths, batch_size=1)

# Iterate over each audio file in the test dataset
for idx, specs in enumerate(tqdm(iter(test_ds), desc='test ', total=len(test_df))):
    # Extract the filename without the extension
    filename = test_paths[idx].split('/')[-1].replace('.ogg','')
    
    # Convert to backend-specific tensor while excluding extra dimension
    specs = keras.ops.convert_to_tensor(specs[0])
    
    # Predict bird species for all frames in a recording using all trained models
    frame_preds = model.predict(specs, verbose=0)
    
    # Create a ID for each frame in a recording using the filename and frame number
    frame_ids = [f'{filename}_{(frame_id+1)*5}' for frame_id in range(len(frame_preds))]
    
    # Concatenate the ids
    ids += frame_ids
    # Concatenate the predictions
    preds = np.concatenate([preds, frame_preds], axis=0)

# %% [markdown]
# # Submission ✉️

# %% [code] {"execution":{"iopub.status.busy":"2024-06-06T11:38:59.121631Z","iopub.execute_input":"2024-06-06T11:38:59.122090Z","iopub.status.idle":"2024-06-06T11:38:59.385055Z","shell.execute_reply.started":"2024-06-06T11:38:59.122054Z","shell.execute_reply":"2024-06-06T11:38:59.384107Z"}}
# Submit prediction
pred_df = pd.DataFrame(ids, columns=['row_id'])
pred_df.loc[:, CFG.class_names] = preds
pred_df.to_csv('submission.csv',index=False)
pred_df.head()
