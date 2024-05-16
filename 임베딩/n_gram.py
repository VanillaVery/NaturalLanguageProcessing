#가장 기초적인 통계적 언어 모델 
# n개의 연속된 단어 시퀀스를 하나의 단위로 취급하여, 특정 단어 시퀀스가 등장할 확률을 추정 
#%%
import nltk

def ngrams(sentence,n):
    words = sentence.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return list(ngrams)

sentence = "오늘도 공부를 늦게 시작했지만 어쨌든 하고 있는 대견한 나."

trigram = ngrams(sentence, 3)
print(trigram)

trigram2 = nltk.ngrams(sentence.split(),3)
print(list(trigram2))

# [('오늘도', '공부를', '늦게'), ('공부를', '늦게', '시작했지만'), ('늦게', '시작했지만', '어쨌든'), ('시작했지만', '어쨌든', '하고'), ('어쨌든', '하고', '있는'), ('하고', '있는', '대견한'), ('있는', '대견한', '나.')]
# [('오늘도', '공부를', '늦게'), ('공부를', '늦게', '시작했지만'), ('늦게', '시작했지만', '어쨌든'), ('시작했지만', '어쨌든', '하고'), ('어쨌든', '하고', '있는'), ('하고', '있는', '대견한'), ('있는', '대견한', '나.')]
