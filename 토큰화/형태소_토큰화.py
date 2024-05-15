"""
KoNLPy
"""
#한국어 자연어 처리를 위해 개발된 라이브러리, 명사 추출, 형태소 분석, 품사 태깅 등의 기능을 제공

#Okt 토큰화
from konlpy.tag import Okt

okt = Okt()

sentence = "무엇이든 상상할 수 있는 사람은 무엇이든 만들어 낼 수 있다"

nouns = okt.nouns(sentence) #명사 추출
phrases = okt.phrases(sentence) #구 추출
morphs = okt.morphs(sentence) #형태소 추출
pos = okt.pos(sentence) #품사 태깅

#%%
"""
NLTK
"""
# 주로 자연어 처리를 위해 개발
import nltk
from nltk import tokenize

nltk.download("punkt") #통계 기반 모델 
nltk.download("averaged_perceptron_tagger") #퍼셉트론 기반 품사 태깅

#"punkt"모델 기반 토큰화
sentence = "Those whe can imagine anything, can create the impossible."

word_tokens = tokenize.word_tokenize(sentence)
sent_tokens = tokenize.sent_tokenize(sentence)

print(word_tokens)
print(sent_tokens)

#%%
# 영문 품사 태깅
from nltk import tag
from nltk import tokenize

# "averaged_perceptron_tagger" 기반 품사 태깅
sentence = "Those whe can imagine anything, can create the impossible."

word_tokens = tokenize.word_tokenize(sentence)
pos = tag.pos_tag(word_tokens)
print(pos)
#%%
"""
spaCy
"""
#nltk와 차이점: 빠른 속도와 높은 정확도를 목표로 하는 머신 러닝 기반의 자연어 처리 라이브러리
#nltk에서 사용하는 모델보다 더 크고 복잡, 더 많은 리소스를 요구
# gpu 가속을 비롯하여, 24개 이상의 언어로 사전 학습된 모델을 제공 

import spacy

nlp = spacy.load("en_core_web_sm") #영어로 사전 학습된 모델
sentence = "Those whe can imagine anything, can create the impossible."
doc = nlp(sentence)

for token in doc:
    print(f"[{token.pos_:5}-{token.tag_:3} : {token.text}]")