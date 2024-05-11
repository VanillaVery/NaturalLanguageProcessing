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
