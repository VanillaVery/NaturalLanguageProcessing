# cnn은 이미지 인식 등 컴퓨터비전 분야의 데이터 분석을 위해 사용되는 인공 신경망
# 입력 데이터의 지역적인 특징을 추출하는 데 특화

# 자연어 처리 작업에서도 우수한 성능을 보임.
#%%
## 합성곱 계층
# 입력 데이터에 필터를 이용해 합성곱 연산을 수행하는 계층
# -> 필터의 가중치가 학습 시 갱신됨

# 연산 시 특징 맵의 크기가 작아지는데, 이것은 깊은 합성곱 신경망을 쌓는데 제약사항이 됨
# 이를 방지하기 위해 패딩을 추가 
#간격을 조절하면 출력 데이터의 크기를 조절 가능
#%%
#합성곱 모델

import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16,kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(32*32*32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
#%%
# 모델 실습
# 텍스트 데이터의 임베딩 값은 순서 말고는 위치가 의미를 가지지 않음...
# 텍스트 데이터는 1차원 합성곱을 적용해야 함 (입력 데이터가 1차원 벡터인 경우에 대한 연산 수행)
#%%
#합성곱 기반 문장 분류 모델 정의 
import torch
from torch import nn

class SentenceClassifier(nn.Module):
    def __init__(self,filter_sizes, max_length,embedding_dim,n_vocab,pretrained_embedding=None, dropout=0.5):
        super().__init__()
        if pretrained_embedding is not None :
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embedding, dtype=torch.float32)
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=n_vocab,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        embedding_dim = self.embedding.weight.shape[1]

        conv = []
        for size in filter_sizes:
            conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=1,
                        kernel_size=size
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=max_length-size-1),
                )
            )
        self.conv_filters = nn.ModuleList(conv)

        output_size = len(filter_sizes)
        self.pre_classifier = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, 1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0,2,1) # 차원을 변경하는 메서드

        conv_outputs = [conv(embeddings) for conv in self.conv_filters]
        concat_outputs = torch.cat([conv.squeeze(-1) for conv in conv_outputs],dim=1)

        logits = self.pre_classifier(concat_outputs)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        return logits
#%%
#합성곱 신경망 분류 모델 학습
# 순환 신경망과 대부분 유사

import pandas as pd
from Korpora import Korpora

corpus = Korpora.load("nsmc")
corpus_df = pd.DataFrame(corpus.test)

train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)

print(train.head(5).to_markdown())
# |       | text                                                                                     |   label |
# |------:|:-----------------------------------------------------------------------------------------|--------:|
# | 33553 | 모든 편견을 날려 버리는 가슴 따뜻한 영화. 로버트 드 니로, 필립 세이모어 호프만 영원하라. |       1 |
# |  9427 | 무한 리메이크의 소재. 감독의 역량은 항상 그 자리에...                                    |       0 |
# |   199 | 신날 것 없는 애니.                                                                       |       0 |
# | 12447 | 잔잔 격동                                                                                |       1 |
# | 39489 | 오랜만에 찾은 주말의 명화의 보석                                                         |       1 |

#%%
# 데이터 토큰화 및 단어 사전 구축
from kiwipiepy import Kiwi
from collections import Counter

def build_vocab(corpus, n_vocab,special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab 

kiwi = Kiwi()
train_tokens =[kiwi.tokenize(review) for review in train.text] #품사 태깅
train_tokens = [[t.form for t in token] for token in train_tokens] #형태소만 가져옴

test_tokens =[kiwi.tokenize(review) for review in test.text] #품사 태깅
test_tokens = [[t.form for t in token] for token in test_tokens] #형태소만 가져옴

vocab = build_vocab(corpus=train_tokens, n_vocab=5000 , special_tokens = ["<pad>","<unk>"])
# 토큰(단어) 사전 구축 완료 
token_to_id = {token:idx for idx,token in enumerate(vocab)}
id_to_token = {id:token for idx,token in enumerate(vocab)}

#%%
#임베딩 층을 사용하기 위해 토큰을 정수로 변환 & 패딩
import numpy as np

def pad_sequences(sequences, max_length, pad_value):
    """
    너무 긴 문장은 최대 길이로 줄이고, 너무 작은 길이라면 최대 길이와 동일한 크기로 변환
    <pad>토큰을 뒤에 붙여 동일한 길이로 변경
    """
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)

unk_id = token_to_id["<unk>"]
train_ids = [[token_to_id.get(tokens,unk_id) for tokens in review] for review in train_tokens]
test_ids = [[token_to_id.get(tokens,unk_id) for tokens in review] for review in test_tokens]

max_length = 32
pad_id = token_to_id["<pad>"]

train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)

print(train_ids)
# array([[ 288, 1553,   18, ...,    0,    0,    0],
#        [2453,  955,   14, ...,    0,    0,    0],
#        [ 360,   23,   20, ...,    0,    0,    0],
#        ...,
#        [1937,   55,   15, ...,    0,    0,    0],
#        [ 125,    0,    0, ...,    0,    0,    0],
#        [ 133,  976,  121, ...,    0,    0,    0]])
#%%
# 데이터로더 적용

import torch
from torch.utils.data import TensorDataset, DataLoader

train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

train_labels = torch.tensor(train.label.values, dtype = torch.float32)
test_labels = torch.tensor(test.label.values, dtype = torch.float32)

train_dataset = TensorDataset(train_ids,train_labels)
test_dataset = TensorDataset(test_ids,test_labels)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=False)

#%%

#손실 함수와 최적화 함수 정의 
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"
filter_sizes = [3,3,4,4,5,5]
n_vocab = len(token_to_id)
classifier = SentenceClassifier(
    pretrained_embedding=None, embedding_dim=128, n_vocab=n_vocab,filter_sizes=filter_sizes, max_length=max_length).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

#%%
#모델 학습 및 테스트
def train(model, datasets, criterion, optimizer, device, interval):
    model.train()
    losses = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0 :
            print(f"Train loss {step} : {np.mean(losses)}")

def test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        yhat = torch.sigmoid(logits)>.5 #결과값이 0.5보다 크면 True or False 출력 
        corrects.extend(
            torch.eq(yhat, labels).cpu().tolist()
        )
    print(f"Val loss: {np.mean(losses)}, Val Accuracy: {np.mean(corrects)}")

epochs = 5
interval = 500

for epoch in range(epochs):
    train(classifier, train_loader, criterion, optimizer, device, interval)
    test(classifier, test_loader, criterion, device)


