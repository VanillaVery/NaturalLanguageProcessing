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
# 기본 SKIP GRAM 클래스
from torch import nn

class VanillaSkipgram(nn.Module):
    def __init__(self,vocab_size , embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embedding = vocab_size,
            embedding_dim = embedding_dim
        )
        self.linear = nn.Linear(
            in_features = embedding_dim,
            out_features =  vocab_size
        )

def foward(self,input_ids):
    embedding = self.embedding(input_ids)
    output = self.linear(embedding)
    return output