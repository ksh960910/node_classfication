from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import model as _model
import trainer as _trainer
import utils as u

torch.manual_seed(u.args.seed)


if __name__ == '__main__':

    dataset = Planetoid(root='/home/user/Stuuddddyyyyyyyyyyyyyyyyyyy', name='Cora')
    data = dataset[0]

    model = _model.GCN(hidden_channels=u.args.hidden, num_features=dataset.num_features, num_classes=dataset.num_classes)
    print(model)

    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Initialize optimizer
    learning_rate = u.args.lr
    # L2 regularization
    decay = u.args.decay

    trainer = _trainer.Trainer(model, data, learning_rate, decay)
    trainer.train_epoch()