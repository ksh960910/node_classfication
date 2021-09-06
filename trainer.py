import torch
import utils as u

class Trainer():

    def __init__(self, model, data, learning_rate, decay):
        self.model = model
        self.data = data
        self.learning_rate = learning_rate
        self.decay = decay
        self.optimizer = torch.optim.Adam(model.parameters(),
                                lr = learning_rate,
                                weight_decay = decay)
        self.criterion = torch.nn.CrossEntropyLoss()                           

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        # 전체 node에 feature가 있기 때문에 전체 data를 사용하여 모델에 forward 시켜줌. 
        out = self.model(self.data.x, self.data.edge_index)
        # loss 값을 구할때는 label이 있는 node들로만 계산
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        pred = out.argmax(dim=1)
        train_correct = pred[self.data.train_mask] == self.data.y[self.data.train_mask]
        train_acc = int(train_correct.sum()) / int(self.data.train_mask.sum())
        self.optimizer.step()
        
        val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
        val_correct = pred[self.data.val_mask] == self.data.y[self.data.val_mask]
        val_acc = int(val_correct.sum()) / int(self.data.val_mask.sum())
        return loss, train_acc, val_loss, val_acc

    def test(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        # 가장 높은 확률로 나온 class 를 pred로 사용
        pred = out.argmax(dim=1)
        # 실제 값과 맞는지 비교
        test_correct = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
        test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())
        return test_acc

        
    def train(self):
        losses = []
        val_losses = []
        for epoch in range(0, u.args.epochs+1):
            loss, train_acc, val_loss, val_acc = self.train_epoch()
            losses.append(loss)
            val_losses.append(val_loss)
            if epoch % 20 == 0:
                print(f'Epoch : {epoch:3d}, Loss : {loss:.4f}, Train accuracy : {train_acc:.4f}, Val_Loss : {val_loss:.4f}, Val accuracy : {val_acc:.4f}')