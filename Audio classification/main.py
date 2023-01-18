import random

import pandas as pd
import os

import torch
import torchaudio
from torch import nn
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import init

path = os.getcwd()
train_path = path + '/train'
test_path = path + '/test'
metadata = pd.read_table(train_path + '/targets.tsv', header=None, names=['audio', 'id'])
for i in range(metadata.shape[0]):
    metadata.at[i, 'audio'] = metadata.at[i, 'audio'] + '.wav'


class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            pad_beg_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - pad_beg_len - sig_len
            beg = torch.zeros((1, pad_beg_len))
            end = torch.zeros((1, pad_end_len))
            sig = torch.cat((beg, sig, end), 1)

        return sig, sr

    @staticmethod
    def spectro_gram(aud, n_mels=18, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_dp = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        return transforms.AmplitudeToDB(top_db=top_dp)(spec)


class SoundDS(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.data_path = path
        self.duration = 6000
        self.sr = 16000
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.at[idx, 'audio']
        label = self.df.at[idx, 'id']
        aud = AudioUtil.open(audio_file)

        reaud = AudioUtil.resample(aud, self.sr)
        dur_aud = AudioUtil.pad_trunc(reaud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud)
        return sgram, label


train_val_dataset = SoundDS(metadata, train_path)
num_items = len(train_val_dataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_dataset, val_dataset = random_split(train_val_dataset, [num_train, num_val])

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
# batch shape: [batch_size=16, 1, n_mels=18, 188]


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 1
        conv_layers = []

        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu, self.bn1]

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu, self.bn2]

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu, self.bn3]

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu, self.bn4]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.output = nn.Linear(64, 2)
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        return self.output(x.flatten(start_dim=1))

    def fit(self, dataset):
        num_epochs = 20
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                        steps_per_epoch=int(len(dataset)),
                                                        epochs=num_epochs,
                                                        anneal_strategy='linear')
        for epoch in range(num_epochs):
            epoch_loss = .0
            correct = 0
            total = 0

            for i, data in enumerate(dataset):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs_m = inputs.mean()
                inputs_s = inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                optimizer.zero_grad()

                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

                _, prediction = torch.max(output, 1)
                correct += (prediction == labels).sum().item()
                total += prediction.shape[0]

            num_batches = len(dataset)
            avg_loss = epoch_loss / num_batches
            acc = correct / total
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Train accuracy: {acc:.2f}')
            if acc >= 0.95:
                torch.save(self, 'model.pt')

        print('Finished training')

    def validate(self, dataset):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(dataset):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs_m = inputs.mean()
                inputs_s = inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                output = model(inputs)
                _, prediction = torch.max(output, 1)
                correct += (prediction == labels).sum().item()
                total += prediction.shape[0]
            print(f'Val accuracy: {correct / total}')


model = AudioClassifier()
device = torch.device('cuda:0')
print(torch.cuda.is_available())
model = model.to(device)
# data, label = next(iter(train_dataloader))
# data, label = data.to(device), label.to(device)
IS_TRAIN = False
if IS_TRAIN:
    model.fit(train_dataloader)
else:
    model = torch.load('model.pt')
model.validate(val_dataloader)

test_data = {'audio': [], 'id': []}
for file in os.listdir(test_path):
    test_data['audio'].append(file)
    test_data['id'].append(0)
test_data = pd.DataFrame(test_data)
test_dataset = SoundDS(test_data, test_path)
test_dataloader = DataLoader(test_dataset, 1, shuffle=False)
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        input_test, _ = data[0].to(device), data[1]
        inputs_m = input_test.mean()
        inputs_s = input_test.std()
        input_test = (input_test - inputs_m) / inputs_s
        output = model(input_test)
        _, prediction = torch.max(output, 1)
        test_data.at[i, 'id'] = prediction.item()
        s = test_data.at[i, 'audio'].replace('.wav', '')
        test_data.at[i, 'audio'] = s
test_data.to_csv('targets.tsv', sep="\t", index=False, header=False)
