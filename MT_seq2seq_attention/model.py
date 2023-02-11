# -*- coding: utf-8 -*-

with open("data.txt") as f:
    data = [l.rstrip().split("\t") for l in f]

print(f"Dataset size {len(data):,}")

## Data preprocessing

import torch
from torch.utils.data import random_split

data_size = len(data)
train_size = int(0.8 * data_size)
test_size = int(0.15 * data_size)
val_size = data_size - train_size - test_size
train_data, test_data, val_data = random_split(
    data, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42)
)
print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")
print(f"Val size: {len(val_data)}")

"""
Here comes the preprocessing.
"""

from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()


def tokenize(sent):
    return tokenizer.tokenize(sent.lower())


from collections import Counter

from torchtext.vocab import vocab as Vocab

src_counter = Counter()
trg_counter = Counter()
for src, trg in train_data:
    src_counter.update(tokenize(src))
    trg_counter.update(tokenize(trg))

src_vocab = Vocab(src_counter, min_freq=3)
trg_vocab = Vocab(trg_counter, min_freq=3)

unk_token = "<unk>"
sos_token, eos_token, pad_token = "<sos>", "<eos>", "<pad>"
specials = [sos_token, eos_token, pad_token]

for vocab in [src_vocab, trg_vocab]:
    if unk_token not in vocab:
        vocab.insert_token(unk_token, index=0)
        vocab.set_default_index(0)

    for token in specials:
        if token not in vocab:
            vocab.append_token(token)

print(f"Source (en) vocabulary size: {len(src_vocab)}")
print(f"Target (ru) vocabulary size: {len(trg_vocab)}")


def encode(sent, vocab):
    tokenized = [sos_token] + tokenize(sent) + [eos_token]
    return [vocab[tok] for tok in tokenized]


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_batch(batch):
    src_list, trg_list = [], []
    for src, trg in batch:
        src_encoded = encode(src, src_vocab)
        src_list.append(torch.tensor(src_encoded))

        trg_encoded = encode(trg, trg_vocab)
        trg_list.append(torch.tensor(trg_encoded))
    src_padded = pad_sequence(src_list, padding_value=src_vocab[pad_token])
    trg_padded = pad_sequence(trg_list, padding_value=trg_vocab[pad_token])
    return src_padded, trg_padded


batch_size = 256
train_dataloader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_data, batch_size // 16, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False, collate_fn=collate_batch)

src_batch, trg_batch = next(iter(train_dataloader))

## Model side

import random

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        states, hidden = self.rnn(embedded)
        return states, hidden


class Decoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        # self.out = nn.Linear(hid_dim, n_tokens)

    def forward(self, input, hidden):
        input = input.unsqueeze(dim=0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        # pred = self.out(output.squeeze(dim=0))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.out = nn.Linear(decoder.hid_dim * 2, decoder.n_tokens)

        assert encoder.hid_dim == decoder.hid_dim, "encoder and decoder must have same hidden dim"
        assert (
                encoder.n_layers == decoder.n_layers
        ), "encoder and decoder must have equal number of layers"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        preds = []
        out_enc, hidden = self.encoder(src)
        out_enc = torch.permute(out_enc, (1, 0, 2))

        # First input to the decoder is the <sos> token.
        input = trg[0, :]
        for i in range(1, trg_len):
            out_dec, hidden = self.decoder(input, hidden)
            out_dec = torch.permute(out_dec, (1, 2, 0))
            attention_score = (out_enc @ out_dec)
            proba = nn.Softmax(dim=1)(attention_score)
            attention_output = torch.sum(out_enc * proba, dim=1)
            # attention_output += out_dec.squeeze() # just sum - BLEU <= 25
            attention_output = torch.cat((attention_output, out_dec.squeeze()), 1)  # concat - BLEU <= 28
            pred = self.out(attention_output)
            preds.append(pred)

            teacher_force = random.random() < teacher_forcing_ratio
            _, top_pred = pred.max(dim=1)
            input = trg[i, :] if teacher_force else top_pred

        return torch.stack(preds)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hid_dim = 512
enc = Encoder(len(src_vocab), emb_dim=256, hid_dim=hid_dim, n_layers=2, dropout=0.5)
dec = Decoder(len(trg_vocab), emb_dim=256, hid_dim=hid_dim, n_layers=2, dropout=0.5)
model = Seq2Seq(enc, dec).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")  # > 21.0 millions

lr = 3e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab[pad_token])
loss_history, train_loss_history, val_loss_history = [], [], []

len(train_dataloader)

import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

# model = torch.load("seq2seq_attention_en_ru.pt")

n_epochs = 25
clip = 1
src, trg, loss, output, train_loss = None, None, None, None, None
for epoch in range(n_epochs):
    if (epoch + 1) % 5 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = 0
    for src, trg in train_dataloader:
        torch.cuda.empty_cache()
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg)

        output = output.view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.detach().item()
        loss_history.append(loss.detach().item())

        if len(loss_history) % 80 == 0:
            plt.figure(figsize=(15, 5))

            plt.subplot(121)
            plt.plot(loss_history)
            plt.xlabel("step")

            plt.subplot(122)
            plt.plot(train_loss_history, label="train loss")
            plt.plot(val_loss_history, label="val loss")
            plt.xlabel("epoch")
            plt.legend()

            plt.show()

    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, trg in val_dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg)

            output = output.view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            val_loss += loss.detach().item()

    val_loss /= len(val_dataloader)
    if epoch >= 5 and val_loss < val_loss_history[-1]:
        torch.save(model, "seq2seq_attention_en_ru.pt")
    if epoch >= 10 and val_loss >= val_loss_history[-1] and val_loss >= val_loss_history[-2]:
        break
    val_loss_history.append(val_loss)

## Model evaluation

trg_itos = trg_vocab.get_itos()
model.eval()
max_len = 50
with torch.no_grad():
    for i, (src, trg) in enumerate(val_data):
        encoded = encode(src, src_vocab)[::-1]
        encoded = torch.tensor(encoded)[:, None].to(device)
        out_enc, hidden = model.encoder(encoded)
        out_enc = torch.permute(out_enc, (1, 0, 2))

        pred_tokens = [trg_vocab[sos_token]]
        for _ in range(max_len):
            decoder_input = torch.tensor([pred_tokens[-1]]).to(device)
            out_dec, hidden = model.decoder(decoder_input, hidden)
            out_dec = torch.permute(out_dec, (1, 2, 0))
            attention_score = (out_enc @ out_dec)
            proba = nn.Softmax(dim=1)(attention_score)
            attention_output = torch.sum(out_enc * proba, dim=1)
            # attention_output += out_dec.squeeze()
            attention_output = torch.cat((attention_output, out_dec.squeeze()), 1)
            pred = model.out(attention_output)

            _, pred_token = pred.max(dim=1)
            if pred_token == trg_vocab[eos_token]:
                # Don't add it to prediction for cleaner output.
                break

            pred_tokens.append(pred_token.item())

        print(f"src: '{src.rstrip().lower()}'")
        print(f"trg: '{trg.rstrip().lower()}'")
        print(f"pred: '{' '.join(trg_itos[i] for i in pred_tokens[1:])}'")
        print()

        if i == 15:
            break

"""
Calc BLEU
"""

from nltk.translate.bleu_score import corpus_bleu

references, hypotheses = [], []
with torch.no_grad():
    for src, trg in test_dataloader:
        output = model(src.to(device), trg.to(device), teacher_forcing_ratio=0)
        output = output.cpu().numpy().argmax(axis=2)

        for i in range(trg.shape[1]):
            reference = trg[:, i]
            reference_tokens = [trg_itos[id_] for id_ in reference]
            reference_tokens = [tok for tok in reference_tokens if tok not in specials]
            references.append(reference_tokens)

            hypothesis = output[:, i]
            hypothesis_tokens = [trg_itos[id_] for id_ in hypothesis]
            hypothesis_tokens = [tok for tok in hypothesis_tokens if tok not in specials]
            hypotheses.append(hypothesis_tokens)

# corpus_bleu works with multiple references
bleu = corpus_bleu([[ref] for ref in references], hypotheses)
print(f"model shows test BLEU of {100 * bleu:.1f}")

# model shows test BLEU of 28.5
