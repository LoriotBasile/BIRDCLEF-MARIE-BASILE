# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:42.458217Z","iopub.execute_input":"2024-06-05T10:22:42.458560Z","iopub.status.idle":"2024-06-05T10:22:44.894511Z","shell.execute_reply.started":"2024-06-05T10:22:42.458494Z","shell.execute_reply":"2024-06-05T10:22:44.893553Z"},"jupyter":{"outputs_hidden":false}}
# Import All Necessary Packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torchaudio
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
from tabulate import tabulate # tabulate print
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore") # Ignore All Warnings

# %% [markdown]
# # Read The csv file.
# ## Print First 5 row

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:44.896074Z","iopub.execute_input":"2024-06-05T10:22:44.896396Z","iopub.status.idle":"2024-06-05T10:22:44.949235Z","shell.execute_reply.started":"2024-06-05T10:22:44.896359Z","shell.execute_reply":"2024-06-05T10:22:44.948419Z"},"jupyter":{"outputs_hidden":false}}
df = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:44.950691Z","iopub.execute_input":"2024-06-05T10:22:44.950978Z","iopub.status.idle":"2024-06-05T10:22:45.199277Z","shell.execute_reply.started":"2024-06-05T10:22:44.950950Z","shell.execute_reply":"2024-06-05T10:22:45.198431Z"},"jupyter":{"outputs_hidden":false}}
wave = torchaudio.load("../input/urbansound8k/fold1/102106-3-0-0.wav")
plt.plot(wave[0].t().numpy())
print(wave[0].shape) # torch.Size([2, 72324]) 2 channels, 72324 sample_rate

# %% [markdown]
# # Process Audio Files That return specgram and label

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:45.200803Z","iopub.execute_input":"2024-06-05T10:22:45.201238Z","iopub.status.idle":"2024-06-05T10:22:53.557911Z","shell.execute_reply.started":"2024-06-05T10:22:45.201204Z","shell.execute_reply":"2024-06-05T10:22:53.557010Z"},"jupyter":{"outputs_hidden":false}}
!pip install noisereduce

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:53.559302Z","iopub.execute_input":"2024-06-05T10:22:53.559618Z","iopub.status.idle":"2024-06-05T10:22:54.782801Z","shell.execute_reply.started":"2024-06-05T10:22:53.559586Z","shell.execute_reply":"2024-06-05T10:22:54.781920Z"},"jupyter":{"outputs_hidden":false}}
import noisereduce as nr

# Importez noisereduce ici
# Ajoutez l'importation de noisereduce juste après les autres importations

class AudioDataset:
    def __init__(self, file_path, class_id):
        self.file_path = file_path
        self.class_id = class_id
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        path = self.file_path[idx]
        waveform, sr = torchaudio.load(path, normalization=True)  # Chargez l'audio
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)  # Convertir en mono
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:  # Normalisation de la longueur
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono = tempData
        
        # Réduction de bruit parasite
        reduced_noise = nr.reduce_noise(audio_mono.numpy(), audio_mono.numpy())
        audio_mono = torch.tensor(reduced_noise)
        
        # Extraction des caractéristiques
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        
        return {
            "specgram": torch.tensor(new_feat[0].permute(1, 0), dtype=torch.float),
            "label": torch.tensor(self.class_id[idx], dtype=torch.long)
        }

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:54.783947Z","iopub.execute_input":"2024-06-05T10:22:54.784228Z","iopub.status.idle":"2024-06-05T10:22:54.794412Z","shell.execute_reply.started":"2024-06-05T10:22:54.784200Z","shell.execute_reply":"2024-06-05T10:22:54.793373Z"},"jupyter":{"outputs_hidden":false}}
class AudioDataset:
    def __init__(self, file_path, class_id):
        self.file_path = file_path
        self.class_id = class_id
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        path = self.file_path[idx]
        waveform, sr = torchaudio.load(path, normalization=True) # load audio
        audio_mono = torch.mean(waveform, dim=0, keepdim=True) # Convert sterio to mono
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000: # if sample_rate < 160000
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000] # else sample_rate 160000
        audio_mono=tempData
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono) # (channel, n_mels, time)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std() # Noramalization
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono) # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std() # mfcc norm
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        
        return {
            "specgram": torch.tensor(new_feat[0].permute(1, 0), dtype=torch.float),
            "label": torch.tensor(self.class_id[idx], dtype=torch.long)
        }

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:54.795788Z","iopub.execute_input":"2024-06-05T10:22:54.796145Z","iopub.status.idle":"2024-06-05T10:22:54.913417Z","shell.execute_reply.started":"2024-06-05T10:22:54.796108Z","shell.execute_reply":"2024-06-05T10:22:54.912686Z"},"jupyter":{"outputs_hidden":false}}
# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(data):
    specs = []
    labels = []
    for d in data:
        spec = d["specgram"].to(device)
        label = d["label"].to(device)
        specs.append(spec)
        labels.append(label)
    spec = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True, padding_value=0.)
    labels = torch.tensor(labels)
    return spec, labels


FILE_PATH = "../input/urbansound8k/"

if __name__ == "__main__":
    df = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")
    files = df["slice_file_name"].values.tolist()
    folder_fold = df["fold"].values
    label = df["classID"].values.tolist()
    path = [
        os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)
    ]
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(path, label, random_state=42, test_size=0.3)
    
    train_dataset = AudioDataset(
        file_path=X_train,
        class_id=y_train
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate_fn
    )
    
    test_dataset = AudioDataset(
        file_path=X_test,
        class_id=y_test
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, drop_last=True, collate_fn=collate_fn
    )

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:54.916192Z","iopub.execute_input":"2024-06-05T10:22:54.916588Z","iopub.status.idle":"2024-06-05T10:22:54.927169Z","shell.execute_reply.started":"2024-06-05T10:22:54.916549Z","shell.execute_reply":"2024-06-05T10:22:54.926221Z"},"jupyter":{"outputs_hidden":false}}
# model
class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(n_hidden), int(n_hidden/2))
        self.fc2 = nn.Linear(int(n_hidden/2), out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)
        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)
        # out.shape (batch, out_feature)
        out = self.fc1(out)
        out = self.fc2(out[:, -1, :])
#         print(out.shape)
        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         print(hidden[0].shape)
        return hidden

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:54.928631Z","iopub.execute_input":"2024-06-05T10:22:54.928949Z","iopub.status.idle":"2024-06-05T10:22:54.952099Z","shell.execute_reply.started":"2024-06-05T10:22:54.928915Z","shell.execute_reply":"2024-06-05T10:22:54.951257Z"},"jupyter":{"outputs_hidden":false}}
AudioLSTM()

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:54.953200Z","iopub.execute_input":"2024-06-05T10:22:54.953556Z","iopub.status.idle":"2024-06-05T10:22:59.262953Z","shell.execute_reply.started":"2024-06-05T10:22:54.953528Z","shell.execute_reply":"2024-06-05T10:22:59.262155Z"},"jupyter":{"outputs_hidden":false}}
# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# %% [markdown]
# # Save Model

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:59.264259Z","iopub.execute_input":"2024-06-05T10:22:59.264645Z","iopub.status.idle":"2024-06-05T10:22:59.268633Z","shell.execute_reply.started":"2024-06-05T10:22:59.264606Z","shell.execute_reply":"2024-06-05T10:22:59.267747Z"},"jupyter":{"outputs_hidden":false}}
def save_model(state, filename):
    torch.save(state, filename)
    print("-> Model Saved")

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:59.269761Z","iopub.execute_input":"2024-06-05T10:22:59.270101Z","iopub.status.idle":"2024-06-05T10:22:59.283322Z","shell.execute_reply.started":"2024-06-05T10:22:59.270072Z","shell.execute_reply":"2024-06-05T10:22:59.282537Z"},"jupyter":{"outputs_hidden":false}}
# Train Set
def train(data_loader, model, epoch, optimizer, device):
    losses = []
    accuracies = []
    labels = []
    preds = []
    model.train()
    loop = tqdm(data_loader) # for progress bar
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.to(device)
        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(data.size(0)))  # Utilisation de la taille du lot actuel
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(output, dim=1)
        winners = probs.argmax(dim=1)
        corrects = (winners == target)
        accuracy = corrects.sum().float() / float(target.size(0))
        accuracies.append(accuracy)
        labels += torch.flatten(target).cpu()
        preds += torch.flatten(winners).cpu()
        loop.set_description(f"EPOCH: {epoch} | ITERATION : {batch_idx}/{len(data_loader)} | LOSS: {loss.item()} | ACCURACY: {accuracy}")
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = sum(losses) / len(losses)
    avg_train_accuracy = sum(accuracies) / len(accuracies)
    report = metrics.classification_report(torch.tensor(labels).numpy(), torch.tensor(preds).numpy())
    print(report)
    return avg_train_loss, avg_train_accuracy

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:59.284470Z","iopub.execute_input":"2024-06-05T10:22:59.284730Z","iopub.status.idle":"2024-06-05T10:22:59.298119Z","shell.execute_reply.started":"2024-06-05T10:22:59.284706Z","shell.execute_reply":"2024-06-05T10:22:59.297394Z"},"jupyter":{"outputs_hidden":false}}
# Test Setting
def test(data_loader, model, optimizer, device):
    model.eval()
    accs = []
    preds = []
    labels = []
    test_accuracies = []
    with torch.no_grad():
        loop = tqdm(data_loader) # Test progress bar
        for batch_idx, (data, target) in enumerate(loop):
            data = data.to(device)
            target = target.to(device)
            output, hidden_state = model(data, model.init_hidden(128))
            probs = torch.softmax(output, dim=1)
            winners = probs.argmax(dim=1)
            corrects = (winners == target)
            accuracy = corrects.sum().float() / float(target.size(0))
            
            test_accuracies.append(accuracy)
            labels += torch.flatten(target).cpu()
            preds += torch.flatten(winners).cpu()
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    return avg_test_acc

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:59.299309Z","iopub.execute_input":"2024-06-05T10:22:59.299683Z","iopub.status.idle":"2024-06-05T10:22:59.314108Z","shell.execute_reply.started":"2024-06-05T10:22:59.299645Z","shell.execute_reply":"2024-06-05T10:22:59.313176Z"},"jupyter":{"outputs_hidden":false}}
# Epoch
EPOCH = 25
OUT_FEATURE = 10 # class
PATIENCE = 5

# %% [markdown]
# ADAM

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T10:22:59.315313Z","iopub.execute_input":"2024-06-05T10:22:59.315715Z","iopub.status.idle":"2024-06-05T11:17:21.778191Z","shell.execute_reply.started":"2024-06-05T10:22:59.315678Z","shell.execute_reply":"2024-06-05T11:17:21.777291Z"},"jupyter":{"outputs_hidden":false}}
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioLSTM(n_feature=168, out_feature=OUT_FEATURE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=PATIENCE)
    
    best_train_acc, best_epoch = 0, 0 # update acc and epoch
    
    for epoch in range(EPOCH):
        avg_train_loss, avg_train_acc = train(train_loader, model, epoch, optimizer, device)
        avg_test_acc = test(test_loader, model, optimizer, device)
        scheduler.step(avg_train_acc)
        if avg_train_acc > best_train_acc:
            best_train_acc = avg_train_acc
            best_epoch = epoch
            filename = f"best_model_at_epoch_{best_epoch}.pth.tar"
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_model(checkpoint, filename)
        
        table = [
            ["avg_train_loss", avg_train_loss], ["avg_train_accuracy", avg_train_acc],
            ["best_train_acc", best_train_acc], ["best_epoch", best_epoch]
        ]
        print(tabulate(table)) # tabulate View
        test_table = [
            ["Avg test accuracy", avg_test_acc]
        ]
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/test', avg_test_acc, epoch)
        print(tabulate(test_table)) # tabulate View

if __name__ == "__main__":
    main() # Run function

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T11:17:21.779411Z","iopub.execute_input":"2024-06-05T11:17:21.779684Z","iopub.status.idle":"2024-06-05T11:17:21.787086Z","shell.execute_reply.started":"2024-06-05T11:17:21.779657Z","shell.execute_reply":"2024-06-05T11:17:21.786160Z"},"jupyter":{"outputs_hidden":false}}
unique_label = dict(zip(df["classID"], df["class"]))
print(unique_label)

# %% [markdown]
# # for single output test

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T11:17:21.788158Z","iopub.execute_input":"2024-06-05T11:17:21.788463Z","iopub.status.idle":"2024-06-05T11:17:21.941357Z","shell.execute_reply.started":"2024-06-05T11:17:21.788433Z","shell.execute_reply":"2024-06-05T11:17:21.940468Z"},"jupyter":{"outputs_hidden":false}}
waveform, sr = torchaudio.load("../input/urbansound8k/fold2/100652-3-0-3.wav")
audio_mono = torch.mean(waveform, dim=0, keepdim=True)
tempData = torch.zeros([1, 160000])
if audio_mono.numel() < 160000:
    tempData[:, :audio_mono.numel()] = audio_mono
else:
    tempData = audio_mono[:, :160000]
audio_mono=tempData
mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
#         print(f'mfcc {mfcc.size()}')
mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
new_feat = torch.cat([mel_specgram, mfcc], axis=1)

data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
new = torch.load("./best_model_at_epoch_0.pth.tar", map_location=torch.device("cpu"))["state_dict"]
model = AudioLSTM(n_feature=168, out_feature=OUT_FEATURE)
model.load_state_dict(new)
model.eval().cpu()
with torch.no_grad():
    for x in data:
        x = x.to("cpu")
        output, hidden_state = model(x, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
        for i, v in unique_label.items():
            if np.argmax(output.numpy()) == i:
                print(f"Predicted Label : {v}")

# %% [markdown]
# # for a fold test

# %% [code] {"execution":{"iopub.status.busy":"2024-06-05T11:21:58.403487Z","iopub.execute_input":"2024-06-05T11:21:58.403831Z","iopub.status.idle":"2024-06-05T11:22:07.651166Z","shell.execute_reply.started":"2024-06-05T11:21:58.403802Z","shell.execute_reply":"2024-06-05T11:22:07.650289Z"},"jupyter":{"outputs_hidden":false}}
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

# Assurez-vous que ce DataFrame existe dans votre environnement
# df = pd.read_csv('../input/urbansound8k/metadata/UrbanSound8K.csv')

PATH = "../input/urbansound8k/fold2"
fold2 = os.listdir(PATH)[:15]

# Assurez-vous de définir ces variables avant d'exécuter le code
# OUT_FEATURE = ...  # nombre de classes de sortie
# unique_label = ... # dictionnaire des labels uniques
# class AudioLSTM(...): ... # définition de votre modèle AudioLSTM

for i in range(len(fold2)):
    class_name = df[df["slice_file_name"] == fold2[i]]['class']
    for item, value in class_name.items():
        print(f"Actual output : {value}")
    
    waveform, sr = torchaudio.load(os.path.join(PATH, fold2[i]))
    audio_mono = torch.mean(waveform, dim=0, keepdim=True)
    tempData = torch.zeros([1, 160000])
    
    if audio_mono.numel() < 160000:
        tempData[:, :audio_mono.numel()] = audio_mono
    else:
        tempData = audio_mono[:, :160000]
    
    audio_mono = tempData
    
    mel_specgram = torchaudio.transforms.MelSpectrogram(sr, power=2)(audio_mono)
    mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
    mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
    
    new_feat = torch.cat([mel_specgram, mfcc], axis=1)
    data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
    
    new = torch.load("./best_model_at_epoch_20.pth.tar", map_location=torch.device("cpu"))["state_dict"]
    model = AudioLSTM(n_feature=168, out_feature=OUT_FEATURE)
    model.load_state_dict(new)
    model.eval().cpu()
    
    with torch.no_grad():
        for x in data:
            x = x.to("cpu")
            output, hidden_state = model(x, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
            for i, v in unique_label.items():
                if np.argmax(output.numpy()) == i:
                    print(f"Predicted Label : {v}")
    
    # Plotting the Mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_specgram.log2()[0, :, :].detach().numpy(), cmap='viridis')
    plt.title('Electrospectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plotting the MFCC
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc[0, :, :].detach().numpy(), cmap='viridis')
    plt.title('MFCC')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.show()
