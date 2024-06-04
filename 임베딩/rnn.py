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