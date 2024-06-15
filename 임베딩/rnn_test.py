import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 RNN 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 10).to(x.device) # 초기 은닉 상태
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) # 마지막 시간 단계의 출력을 사용
        return out

# 모델 생성, 손실 함수와 옵티마이저 정의
model = SimpleRNN(input_size=1, hidden_size=10, output_size=1, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 데이터
inputs = torch.FloatTensor([[[1], [2], [3], [4]]]) # 형태: (1, 4, 1)
target = torch.FloatTensor([[5]]) # 형태: (1, 1)

# 학습 루프
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 테스트
model.eval()
with torch.no_grad():
    test_input = torch.FloatTensor([[[2], [3], [4], [5]]])
    test_output = model(test_input)
    print(f'Predicted next number: {test_output.item()}')
