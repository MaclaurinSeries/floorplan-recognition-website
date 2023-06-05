import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torchmetrics import Accuracy

from torch_geometric.nn.models import GCN


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, batch_size: int,
                 hidden_channels: int = 256, num_layers: int = 4,
                 dropout: float = 0.5, learning_rate: float = 1e-5):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        super().__init__()
        self.gnn = GCN(in_channels, hidden_channels, num_layers,
                        out_channels, dropout=dropout,
                        norm=BatchNorm1d(hidden_channels))

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)


    def forward(self, x, edge_index, edge_attr):
        y_hat = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        return y_hat


    def training_step(self, batch, batch_index):
        x, y = batch.x, batch.y
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        y_hat = self.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.train_acc(y_hat.softmax(dim=-1), y.argmax(dim=-1))

        self.log_dict({
            'acc': accuracy,
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return loss


    def validation_step(self, batch, batch_index):
        x, y = batch.x, batch.y
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        y_hat = self.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        pred = y_hat.softmax(dim=-1)

        return y_hat, pred, y
    

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in validation_step_outputs:
            val_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
            num_total += pred.shape[0]

        val_accuracy = num_correct / num_total
        val_loss = val_loss / num_total

        self.log_dict({
            "acc/val": val_accuracy,
            "loss/val": val_loss
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)


    def on_validation_end(self) -> None:
        print()


    def test_step(self, batch, batch_index):
        x, y = batch.x, batch.y
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        y_hat = self.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        pred = y_hat.softmax(dim=-1)

        return y_hat, pred, y

    
    def test_epoch_end(self, test_step_outputs):
        test_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in test_step_outputs:
            test_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
            num_total += pred.shape[0]

        test_accuracy = num_correct / num_total
        test_loss = test_loss / num_total

        self.log_dict({
            "acc/test": test_accuracy,
            "loss/test": test_loss
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)


    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        y_hat = self.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        pred = y_hat.softmax(dim=-1)

        return pred


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


