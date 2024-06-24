# 순환 신경망, 합성곱 신경망과 달리 어텐션 메커니즘 만을 사용하여 시퀀스 임베딩을 표현
# 순환 신경망과 달리 입력 시퀀스를 병렬 구조로 처리하므로, 단어의 순서 정보를 제공하지 않음
#%%
# 위치 인코딩
import math
import torch
from torch import nn
from matplotlib import pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout = 0.1): 
        """
        d_model : 입력 임베딩 차원
        max_len : 최대 시퀀스
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model)
            )
        
        pe = torch.zeros(max_len,1,d_model)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:,x.size(0)]
        return self.dropout(x)
    

