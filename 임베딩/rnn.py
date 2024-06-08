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

#%%
# 데이터 토큰화 및 단어 사전 구축
