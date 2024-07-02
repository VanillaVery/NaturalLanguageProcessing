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
    
#%%
# 모델 실습: 파이토치에서 제공하는 트랜스포머 모델을 활용해 영어-독일어 번역 모델을 구성
# Multi30k 데이터세트: 영-독 병렬 말뭉치

# 데이터세트 다운로드 및 전처리
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def generate_tokens(text_iter, language):
    """
    텍스트 반복자(text_iter)와 언어(language)를 입력으로 받아 텍스트를 토큰화된 형태로 반환

    """
    language_index = {SRC_LANGUAGE : 0, TGT_LANGUAGE: 1}

    for text in text_iter:
        yield token_transform[language](text[language_index[language]])

#독일어 말뭉치와 영어 말뭉치에 대해 각각 토크나이저와 어휘사전을 생성
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2,3
special_symbols = ["<unk>","<pad>","<bos>","<eos>"]

#get_tokenizer: 사용자가 지정한 토크나이저를 가져오는 함수
token_transform = {
    SRC_LANGUAGE:get_tokenizer("spacy",language="de_core_news_sm"),
    TGT_LANGUAGE:get_tokenizer("spacy",language="en_core_web_sm"),
}


print("Token Transform:")
print(token_transform)

#vocab_transform:토큰을 인덱스로 변환시키는 함수를 저장
vocab_transform = {}
for language in [SRC_LANGUAGE,TGT_LANGUAGE]:
    train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE,TGT_LANGUAGE))
    vocab_transform[language] = build_vocab_from_iterator(
        generate_tokens(train_iter,language),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )

for language in [SRC_LANGUAGE,TGT_LANGUAGE]:
    vocab_transform[language].set_default_index(UNK_IDX)

print("Vocab Transform:")
print(vocab_transform)
# 아이디어를 정리