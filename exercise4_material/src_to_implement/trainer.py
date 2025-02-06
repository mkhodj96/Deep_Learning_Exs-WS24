from cProfile import label
import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm



class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        """Performs a single training step."""
        self.optim.zero_grad()
        output = self.model(x)
        loss = self.crit(output, y.float())
        loss.backward()
        self.optim.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        
        """Performs a validation_step."""
        output = self.model(x)
        loss = self.crit(output, y.float())

        output_np = output.detach().cpu().numpy()
        pred = (output_np > 0.5).astype(int)

        return loss.item(), pred
        
    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0

        for x, y in self.train_dl:
            if self.cuda:
                x, y = x.cuda(), y.cuda()

            loss = self.train_step(x, y)
            total_loss += loss / len(self.train_dl)

        return total_loss
    
    def val_test(self):
        """Evaluates the model on the validation/test set."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with t.no_grad():
            for x, y in self.val_test_dl:
                if self.cuda:
                    x, y = x.cuda(), y.cuda()

                loss, preds = self.val_test_step(x, y)
                total_loss += loss / len(self.val_test_dl)

                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        f1 = f1_score(np.array(all_labels), np.array(all_preds), average='micro')
        return total_loss, f1
    
        
def fit(self, epochs=-1):
    """Trains the model while applying early stopping if necessary."""
    assert self.early_stopping_patience > 0 or epochs > 0

    train_losses, val_losses, val_metrics = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_n = 0  

    while True:
        if epochs > 0 and epoch_n >= epochs:
            break 
        
        print(f"Epoch {epoch_n + 1}/{epochs if epochs > 0 else '∞'}")

        train_loss = self.train_epoch()
        val_loss, val_metric = self.val_test()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        print(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 Score: {val_metric:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self.save_checkpoint()
            patience_counter = 0  
        else:
            patience_counter += 1

        # Early stopping
        if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
            print("Early stopping triggered.")
            break

        epoch_n += 1 

    return train_losses, val_losses, val_metrics