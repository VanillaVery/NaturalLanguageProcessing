# 하나의 단어가 빈번하게 사용되는 하위 단어의 조합으로 나누어 토큰화하는 방법

# 센텐스피스 라이브러리와 코포라 라이브러리를 활용해 토크나이저를 학습
"""
센텐스피스 라이브러리
"""
#하위 단어 토크나이저 라이브러리

"""
코포라 라이브러리
"""
# 국립국어원이나 AI허브에서 제공하는 말뭉치 데이터를 쉽게 사용할 수 있게 제공하는 오픈소스 라이브러리
#%%
#토크나이저 모델 학습
from Korpora import Korpora

corpus = Korpora.load("korean_petitions") # 청와대 청원 말뭉치
dataset = corpus.train
# KoreanPetitions.train: size=433631
#   - KoreanPetitions.train.texts : list[str]
#   - KoreanPetitions.train.categories : list[str]
#   - KoreanPetitions.train.num_agrees : list[int]
#   - KoreanPetitions.train.begins : list[str]
#   - KoreanPetitions.train.ends : list[str]
#   - KoreanPetitions.train.titles : list[str]
petition = dataset[0]
# KoreanPetition(text="안녕하세요. 현재 사대, 교대 등 교원양성학교들의 예비교사들이 임용절벽에 매우 힘들어 하고 있는 줄로 압니다. 정부 부처에서는 영양사의 영양'교사'화, 폭발적인 영양'교사' 채용, 기간제 교사, 영전강, 스강의 무기계약직화가 그들의 임용 절벽과는 전혀 무관한 일이라고 주장하고 있지만 조금만 생각해보면 전혀 설득력 없는 말이라고 생각합니다. 학교 수가 같고, 학생 수가 동일한데 영양교사와 기간제 교사, 영전강 스강이 학교에 늘어나게 되면 당연히 정규 교원의 수는 줄어들게 되지 않겠습니까? 기간제 교사, 영전강, 스강의 무기계약직화, 정규직화 꼭 전면 백지화해주십시오. 백년대계인 국가의 교육에 달린 문제입니다. 단순히 대통령님의 일자리 공약, 81만개 일자리 창출 공약을 지키시고자 돌이킬 수 없는 실수는 하지 않으시길 바랍니다. 세계 어느 나라와 비교해도, 한국 교원의 수준과 질은 최고 수준입니다. 고등교육을 받고 어려운 국가 고시를 통과해야만 대한민국 공립 학교의 교단에 설 수 있고, 이러한 과정이 힘들기는 하지만 교원들이 교육자로서의 사명감과 자부심을 갖고 교육하게 되는 원동력이기도 합니다. 자격도 없는 비정규 인력들을 일자리 늘리기 명목 하에 학교로 들이게 되면, 그들이 무슨 낯으로 대한민국이 '공정한 사회' 라고 아이들에게 가르칠 수 있겠습니까? 그들이 가르치는 것을 학부모와 학생들이 납득할 수 있겠으며, 학생들은 공부를 열심히 해야하는 이유를 찾을 수나 있겠습니까? 열심히 안 해도 떼 쓰면 되는 세상이라고 생각하지 않겠습니까? 영양사의 영양교사화도 재고해주십시오. 영양사분들 정말 너무나 고마운 분들입니다. 학생들의 건강과 영양? 당연히 성장기에 있는 아이들에게 필수적이고 중요한 문제입니다. 하지만 이들이 왜 교사입니까. 유래를 찾아 볼 수 없는 영양사의 '교사'화. 정말 대통령님이 생각하신 아이디어라고 믿기 싫을 정도로 납득하기 어렵습니다. 중등은 실과교과 교사가 존재하지요? 초등 역시 임용 시험에 실과가 포함돼 있으며 학교 현장에서도 정규 교원이 직접 실과 과목을 학생들에게 가르칩니다. 영양'교사', 아니 영양사가 학생들에게 실과를 가르치지 않습니다. 아니 그 어떤 것도 가르치지 않습니다. 올해 대통령님 취임 후에 초등, 중등 임용 티오가 초전박살 나는 동안 영양'교사' 티오는 폭발적으로 확대된 줄로 압니다. 학생들의 교육을 위해 정말 교원의 수를 줄이고, 영양 교사의 수를 늘리는 것이 올바른 해답인지 묻고 싶습니다. 마지막으로 교원 당 학생 수. 이 통계도 제대로 내주시기 바랍니다. 다른 나라들은 '정규 교원', 즉 담임이나 교과 교사들로만 통계를 내는데(너무나 당연한 것이지요) 왜 한국은 보건, 영양, 기간제, 영전강, 스강 까지 다 포함해서 교원 수 통계를 내는건가요? 이런 통계의 장난을 통해 OECD 평균 교원 당 학생 수와 거의 비슷한 수준에 이르렀다고 주장하시는건가요? 학교는 교육의 장이고 학생들의 공간이지, 인력 센터가 아닙니다. 부탁드립니다. 부디 넓은 안목으로 멀리 내다봐주시길 간곡히 부탁드립니다.", category='육아/교육', num_agree=88, begin='2017-08-25', end='2017-09-24', title='학교는 인력센터, 취업센터가 아닙니다. 정말 간곡히 부탁드립니다.')
#%%
#학습 데이터세트 생성
from Korpora import Korpora

petitions = corpus.get_all_texts() #본문 데이터세트를 한 번에 불러옴

#청원 데이터를 하나의 텍스트 파일로 저장
with open("../datasets/corpus.txt","w",encoding='utf-8') as f:
    for petition in petitions:
        f.write(petition+"\n")
#%%
# 파일을 활용해 토크나이저 모델 학습
from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.Train(
    "--input=../datasets/corpus.txt \
    --model_prefix=petition_bpe \
    --vocab_size=8000 model_type=bpe"
)

#%%
#모델과 어휘 사전 파일을 활용한 바이트 페어 인코딩 수행
from sentencepiece import SentencePieceProcessor

tokenizer = SentencePieceProcessor()
tokenizer.load("petition_bpe.model")

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["너무 늦게 공부를 시작해서","굉장히 졸리고 자고싶군요?","하지만 이건 끝내고 자겠습니다"]

tokenize_sentence = tokenizer.encode_as_pieces(sentence) #문장 토큰화
tokenize_sentences = tokenizer.encode_as_pieces(sentences)

print("단일 문장 토큰화:",  tokenize_sentence)
print("여러 문장 토큰화:", tokenize_sentences)

encoding_sentence = tokenizer.encode_as_ids(sentence) #토큰을 정수로 인코딩 / 정수=토큰에 매핑된 id값
encoding_sentences = tokenizer.encode_as_ids(sentences)

print("단일 문장 정수 인코딩:",  encoding_sentence)
print("여러 문장 정수 인코딩:", encoding_sentences)

decode_ids = tokenizer.decode_ids(encoding_sentences)
decode_pieces = tokenizer.decode_pieces(encoding_sentences)

print("정수 인코딩에서 문장 변환:",decode_ids)
print("하위 단어 토큰에서 문장 변환:",decode_pieces)


# 단일 문장 토큰화: ['▁안녕하세요', ',', '▁토', '크', '나', '이', '저', '가', '▁잘', '▁학', '습', '되었', '군요', '!']
# 여러 문장 토큰화: [['▁너무', '▁늦게', '▁공부를', '▁시작', '해서'], ['▁굉장히', '▁졸', '리고', '▁자', '고싶', '군요', '?'], ['▁하지만', '▁이건', '▁끝', '내고', '▁자', '겠습니다']]
# 단일 문장 정수 인코딩: [667, 6553, 994, 6880, 6544, 6513, 6590, 6523, 161, 110, 6554, 872, 787, 6648]
# 여러 문장 정수 인코딩: [[172, 5072, 3957, 828, 91], [4238, 1756, 120, 32, 2826, 787, 6581], [280, 1012, 731, 1514, 32, 252]]
# 정수 인코딩에서 문장 변환: ['너무 늦게 공부를 시작해서', '굉장히 졸리고 자고싶군요?', '하지만 이건 끝내고 자겠습니다']
# 하위 단어 토큰에서 문장 변환: ['너무 늦게 공부를 시작해서', '굉장히 졸리고 자고싶군요?', '하지만 이건 끝내고 자겠습니다']
#%%
# 어휘 사전 불러오기
from sentencepiece import SentencePieceProcessor

tokenizer = SentencePieceProcessor()
tokenizer.load("petition_bpe.model")

vocab = {idx:tokenizer.id_to_piece(idx) for idx in range(tokenizer.get_piece_size())}
print(list(vocab.items())[:5])
#%%
#토크나이저스
#허깅 페이스의 토크나이저스 라이브러리 이용 / 정규화와 사전 토큰화 제공

#정규화: 불필요한 공백 제고, 대소문자 변환, 유니코드 정규화 등등
#사전 토큰화: 입력 문장을 토큰화하기 전에 단어와 같은 작은 단위로 나누는 기능 제공

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence,NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece())
tokenizer.nomalizer = Sequence([NFD(),Lowercase()]) # 유니코드 정규화, 소문자 변환
tokenizer.pre_tokenizer = Whitespace() #공백, 구두점 기준 분리

tokenizer.train(['../datasets/corpus.txt'])
tokenizer.save("../model/petition_wordpiece.json")
#%%
#워드피스 토큰화
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer.from_file("../model/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()

sentence = "아니, 4장만 하면 되는 공부를 몇시간째 하고 있는거야"
sentences = ["아니야, 스스로를 비난하지 말자!","꾸역꾸역 하고 있는 게 얼마나 대견해!?"]

encoded_sentence = tokenizer.encode(sentence)
print(encoded_sentence.tokens)
# ['아니', ',', '4', '##장', '##만', '하면', '되는', '공부를', '몇', '##시간', '##째', '하고', '있는거', '##야']

encoded_sentences =tokenizer.encode_batch(sentences)
print([enc.tokens for enc in encoded_sentences])
# [['아니', '##야', ',', '스스로', '##를', '비난', '##하지', '말', '##자', '!'],
#  ['꾸', '##역', '##꾸', '##역', '하고', '있는', '게', '얼마나', '대', '##견', '##해', '!', '##?']]
