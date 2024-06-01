# 임베딩 모델 : 단어 간의 유사성을 측정하기 위해 분포 가설을 기반으로 개발됨
# 분포 가설: 같은 문맥에서 함께 자주 나타나는 단어들은 서로 유사한 의미를 가질 가능성이 높다는 가정
 
# 단어 벡터화: 희소 표현과 밀집 표현으로 나눔
#희소 표현: 원-핫 인코딩, tf-idf / word2vec은 밀집 표현 

#word2vec은 밀집 표현을 위해 CBoW와 Skip-gram이라는 두 가지 방법을 사용 

#%%
#네거티브 샘플링 :전체 단어 집합에서 일부 단어를 샘플링하여 오답 단어로 사용

"""# 모델 실습: Skip-Gram"""
# 단어의 수 V, 임베딩 차원을 E -> W(V,E) , W'(E,V)를 최적화 하며 학습
# 임베딩 클래스를 사용하여 룩업 계층을 구현해보자

#%%
# 기본 SKIP GRAM 클래스(모델 선언)
from torch import nn

class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size , embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim
        )
        self.linear = nn.Linear(
            in_features = embedding_dim,
            out_features =  vocab_size
        )

    def forward(self,input_ids):
        embedding = self.embedding(input_ids) # lookup vector
        output = self.linear(embedding)
        return output # 내적값

#%%
# 모델 학습에 사용할 데이터세트 로드, 코포라 라이브러리의 네이버 영화 리뷰 감정 분석 데이터세트 
import pandas as pd
from Korpora import Korpora
# from konlpy.tag import Mecab 너무나 힘들다
from kiwipiepy import Kiwi

corpus = Korpora.load("nsmc")
corpus = pd.DataFrame(corpus.test)

kiwi = Kiwi()
tokens =[kiwi.tokenize(review) for review in corpus.text] #품사 태깅
morphs = [[t.form for t in token] for token in tokens] #형태소만 가져옴
print(morphs[:3]) # 수정 완료!
#%%
#단어 사전 구축
from collections import Counter

def build_vocab(corpus, n_vocab, special_tokens):
    """
    corpus: 말뭉치를 형태소로 나눈 리스트
    n_vocab: 단어 사전 길이
    special_tokens = 특별한 의미를 갖는 토큰 (unk: oov에 대응하기 위한 토큰- 단어 사전 내에 없는 모든 단어는 <unk>토큰으로 대체)
    """
    counter = Counter()
    for tokens in corpus: #말뭉치에 대해 순회
        counter.update(tokens) # 단어의 빈도 업데이트
    vocab = special_tokens
    for token, count in counter.most_common(n_vocab): #빈도 상위 n_vocab개의 단어, 빈도
        vocab.append(token) # 특별 토큰 + 상위 5000개 단어 
    return vocab    
    
vocab = build_vocab(corpus = morphs, n_vocab = 5000, special_tokens=["<unk>"])
tokens_to_id = {token:idx for idx, token in enumerate(vocab)} # 토큰과 인덱스 매칭
id_to_token = {idx:token for idx, token in enumerate(vocab)} #인덱스와 토큰 매칭

print(vocab[:10])
print(len(vocab))
#%%
# skip-gram의 단어 쌍 추출
# 중심 단어의 앞, 뒤 두개 단어를 윈도로 하여 단어쌍 추출
def get_word_pairs(tokens, window_size):
    """
    morphs를 입력받아 skip-gram의 입력 데이터로 사용할 수 있게 전처리
    중심 단어와 주변 단어의 쌍 생성
    """
    pairs = []
    for sentence in tokens: #형태소 리스트를 순회
        sentence_length = len(sentence) #해당 문장의 형태소 길이
        for idx, center_word in enumerate(sentence): #각각의 형태소를 중심 단어로 하여 순회
            window_start = max(0,idx - window_size) # 문장의 경계 넘어가지 않게 조정 
            window_end = min(sentence_length,idx + window_size + 1)
            center_word = sentence[idx]
            context_words = sentence[window_start:idx] + sentence[idx+1:window_end]
            for context_word in context_words: #주변 단어 순회
                pairs.append([center_word,context_word]) # 중심 단어, 주변단어 1 pairing
    return pairs

word_pairs = get_word_pairs(morphs, window_size = 2)
print(word_pairs[:10])
# [['굳', 'ㅋ'], ['ㅋ', '굳'], ['뭐', '이'], ['뭐', '야'], ['이', '뭐'], ['이', '야'], ['이', '이'], ['야', '뭐'], ['야', '이'], ['야', '이']]
#%%
# 임베딩 층이 단어의 인덱스를 입력으로 받기 때문에 단어쌍을 인덱스 쌍으로 변환
def get_index_pairs(word_pairs,tokens_to_id):
    pairs = []
    unk_index = tokens_to_id["<unk>"]
    for word_pair in word_pairs: # 주변 단어, 중심 단어 쌍을 순회
        center_word, context_word = word_pair
        center_index = tokens_to_id.get(center_word,unk_index) # 해당 토큰이 있으면 인덱스 반환, 없으면 unk 인덱스 반환
        context_index = tokens_to_id.get(context_word,unk_index) # 해당 토큰이 있으면 인덱스 반환, 없으면 unk 인덱스 반환
        pairs.append([center_index,context_index])
    return pairs

index_pairs = get_index_pairs(word_pairs,tokens_to_id)
print(index_pairs[:5]) #해당 인덱스 쌍은 skipgram 모델의 입력 데이터로 사용됨
#%%
#인덱스 쌍을 데이터로더에 적용 
import torch
from torch.utils.data import TensorDataset, DataLoader

index_pairs = torch.tensor(index_pairs)
center_indexes = index_pairs[:,0]
context_indexes = index_pairs[:,1]

dataset = TensorDataset(center_indexes,context_indexes)
dataloader = DataLoader(dataset, batch_size = 32,shuffle=True)
#%%
#skip-gram 모델 준비 작업
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"
word2vec = VanillaSkipgram(vocab_size= len(tokens_to_id),embedding_dim=128).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(word2vec.parameters(),lr=0.1)
#%%
#모델 학습
from tqdm import tqdm
for epoch in tqdm(range(10)):
    cost = 0.0
    for input_ids, target_ids in dataloader: #배치
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = word2vec(input_ids)
        # print(logits)
        loss = criterion(logits,target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost+=loss
    print(logits)
    cost = cost / len(dataloader)
    print(f"Epoch : {epoch+1:4d}, Cost : {cost:.3f}")

#%%
#임베딩 값 추출

token_to_embedding =dict()
embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()

for word, embedding in zip(vocab, embedding_matrix):
    token_to_embedding[word] = embedding

index = 30
token = vocab[30]
token_embedding = token_to_embedding[token]
print(token)
print(token_embedding)
#%%
#단어 임베딩 유사도 계산
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a,b):
    """
    입력 단어와 단어 사전 내의 모든 단어와의 코사인 유사도 계산
    a: 임베딩 토큰 -> [,128]
    b: 임베딩 행렬 -> [N,128]이므로 노름 계산시 axis=1 방향으로 계산
    """
    cosine = np.dot(b,a)/(norm(b,axis=1)*norm(a))
    return cosine

def top_n_index(cosine_matrix, n):
    closest_indexes = cosine_matrix.argsort()[::-1]
    top_n = closest_indexes[1: n+1]
    return top_n

cosine_matrix = cosine_similarity(token_embedding,embedding_matrix)
top_n = top_n_index(cosine_matrix,n=5)

