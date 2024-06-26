# rnn 모델은 순서가 있는 연속적인 데이터를 처리하는 데 적합한 구조를 가지고 있다.
# 이전에 처리한 데이터 + 현재 입력 데이터 -> 현재 상태를 예측하는 데 사용
# 다양한 구조로 모델을 설계 가능-> 단순 순환 구조, 일대다 구조, 다대일 구조, 다대다 구조
#%%
# 일대다 구조
# 하나의 입력 -> 여러개의 출력값 / 문장 입력시, 품사 예측 or 이미지 데이터-> 설명 출력 등 
# 이를 위해서는 출력 구조를 미리 알고 있어야 하고, 시퀀스 정보를 활용해 출력 시퀀스의 길이를 예측하는 모델을 함께 구현해야 함
#%%
# 다대일 구조: 여러개의 입력-> 하나의 출력값 / 감성 분류, 문장 분류, 자연어 추론 등
#%%
# 양방향 다층 신경망
import torch
from torch import nn

input_size = 128
output_size = 256
num_layers = 3
bidirectional = True

model = nn.RNN(
    input_size=input_size,
    hidden_size=output_size,
    num_layers=num_layers,
    nonlinearity="tanh",
    batch_first=True,
    bidirectional=bidirectional,
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size) #정규 분포를 따르는 난수 생성
#torch.Size([6, 4, 256])
h_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size,output_size) # 0~1 사이의 값 균등생성
#[계층 수x양방향 여부 +1, 배치 크기, 은닉 상태 크기]의 형태로 구성됨
#torch.Size([6, 4, 256])
outputs, hidden = model(inputs, h_0)
print(outputs.shape) #torch.Size([4, 6, 512])
print(hidden.shape) #torch.Size([6, 4, 256])
#%%
#LSTM
import torch
from torch import nn
lstm = torch.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    bias=False,
    batch_first=True,
    dropout=0,
    bidirectional=False,
    proj_size=0 # >0일 시, 출력차원을 줄이거나 변환 가능 
)

#%%
# 다양한 구조의 lstm 구성 가능
# 양방향 다층 장단기 메모리 구성 

import torch
from torch import nn

input_size = 128
output_size = 256
num_layers = 3
bidirectional = True
proj_size = 64

model = nn.LSTM(
    input_size=input_size,
    hidden_size=output_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    proj_size=proj_size
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size)
h_0 = torch.rand(
    num_layers * (int(bidirectional) + 1),
    batch_size,
    proj_size if proj_size > 0 else output_size
)
c_0 = torch.rand(num_layers * (int(bidirectional) +1), batch_size, output_size)

outputs, (h_n,c_n) = model(inputs, (h_0,c_0))

print(outputs.shape) # torch.Size([4, 6, 128])
print(h_n.shape) # torch.Size([6, 4, 64])
print(c_n.shape) # torch.Size([6, 4, 256])

#%%
# 모델 실습
# rnn, lstm을 이용한 문장 긍/부정 분류

#문장 분류 모델 
import torch
from torch import nn

class SentenceClassifier(nn.Module):
    def __init__(
            self,
            n_vocab,
            hidden_dim,
            embedding_dim,
            n_layers,
            dropout = 0.5,
            bidirectional = True,
            model_type ="lstm"
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        if model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional = bidirectional,
                dropout=dropout,
                batch_first=True 
            )
        elif model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        if bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2,1)
        else:
            self.classifier = nn.Linear(hidden_dim ,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:,-1,:]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits


# model = SentenceClassifier(
#     n_vocab=10000,
#     hidden_dim=128,
#     embedding_dim=300,
#     n_layers=2,
#     dropout=0.5,
#     bidirectional=True,
#     model_type="lstm",

# )
# inputs = torch.randint(10000, (32, 50))  # (batch_size, sequence_length)
# logits = model(inputs)
# print(logits.shape)
#%%
#데이터 세트 불러오기 
# 네이버 영화 리뷰 감정 분석 데이터세트 
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

n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = SentenceClassifier(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim = embedding_dim, n_layers = n_layers).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)

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


#%%
# 학습된 모델로부터 단어 사전의 임베딩 추출
token_to_embedding = dict()
embedding_matrix = classifier.embedding.weight.detach().numpy()

for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb

token = vocab[1000]
print(token, token_to_embedding[token])
#그러나 긍 부정 분류의 경우 임베딩 계층이 아닌 순환 신경망의 연산이 더 중요하게 동작하므로, 
#모델이 복잡할수록 임베딩 계층이 토큰의 의미 정보를 학습하기는 더 어렵다.