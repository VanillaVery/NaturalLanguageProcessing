#텍스트 데이터를 의미를 갖는 최소 단위로 분해
#%%
"""
단어 토큰화
"""
# 띄어쓰기, 문장 부호, 대소문자 등의 특정 구분자를 활용해 토큰화가 수행
review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
tokenized = review.split()

# ['현실과',
#  '구분',
#  '불가능한',
#  'cg.',
#  '시각적',
#  '즐거움은',
#  '최고!',
#  '더불어',
#  'ost는',
#  '더더욱',
#  '최고!!']

# oov 문제, 접사나 부호, 오타 등에 취약
#%%
"""
글자 토큰화
"""
#글자 단위로 문장을 나눔 / 비교적 작은 단어 사전 구축
# 언어 모델링과 같은 시퀀스 예측 작업에서 활용 

review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
tokenized = list(review)
# ['현',
#  '실',
#  '과',
#  ' ',
#  '구',
#  '분',
#  ' ',
#  '불',
#  '가',
#  '능',
#  '한',
#  ' ',
#  'c',
#  'g',
#  '.',
#  ' ',
#  '시',
#  '각', ...
#%%
"""
자모 단위 토큰화
"""
import jamo
from jamo import h2j, j2hcj

review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
decomposed = j2hcj(h2j(review))
list(decomposed)
# ['ㅎ',
#  'ㅕ',
#  'ㄴ',
#  'ㅅ',
#  'ㅣ',
#  'ㄹ',
#  'ㄱ',
#  'ㅘ',
#  ' ',
#  'ㄱ',
#  'ㅜ',
#  'ㅂ',
#  'ㅜ',

#적은 크기의 단어 사전 구축 가능
#개별 토큰은 아무 의미가 없음
