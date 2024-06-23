"""
cnn 을 이용한 문장 분류 모델의 내부 동작 파악 
"""
#%% cnn 파일에서 정의한 input x, label
x, y = train_ids,train_labels

#%%
import torch
from torch import nn

class SentenceClassifier(nn.Module):
    def __init__(self,filter_sizes, max_length,embedding_dim,n_vocab,pretrained_embedding=None, dropout=0.5):
        super().__init__()
        if pretrained_embedding :
            embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embedding, dtype=torch.float32)
            )
        else:
            embedding = nn.Embedding(
                num_embeddings=n_vocab + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        embedding_dim = embedding.weight.shape[1]

        conv = []
        for size in filter_sizes:
            conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=1,
                        kernel_size=size
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=max_length-size-1),
                )
            )
        conv_filters = nn.ModuleList(conv)

        
        output_size = len(filter_sizes)
        pre_classifier = nn.Linear(output_size, output_size)
        dropout = nn.Dropout(dropout)
        classifier = nn.Linear(output_size, 1)

    def forward(self, inputs):
        embeddings = embedding(inputs) #input: [32], output: [32,128]
        embeddings = embeddings.unsqueeze(0) 
        #하나의 문장에 대해 보고 있으므로 차원 변경 , torch.Size([1, 32, 128])
        embeddings = embeddings.permute(0,2,1) 
        # 차원을 변경, torch.Size([1, 128, 32]) 
        # # 차원을 왜 변경? conv1d가 원하는 크기로 변환

        conv_outputs = [conv(embeddings) for conv in conv_filters]
        concat_outputs = torch.cat([conv.squeeze(-1) for conv in conv_outputs],dim=1)

        logits = pre_classifier(concat_outputs)
        logits = dropout(logits)
        logits = classifier(logits)
        return logits
