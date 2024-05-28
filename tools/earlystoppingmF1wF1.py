import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc = 0
        self.mF1 = 0
        self.wF1 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc, mF1, wF1, model, modelname, str):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.acc = acc
            self.mF1 = mF1
            self.wF1 = wF1
            self.save_checkpoint(val_loss, model, modelname, str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print('BEST LOSS:{:.4f}|Accuracy: {:.4f}|mF1:{:.4f}|wF1:{:.4f}'.format(-self.best_score, self.acc,
                                                                                       self.mF1, self.wF1))
        else:
            self.best_score = score
            self.acc = acc
            self.mF1 = mF1
            self.wF1 = wF1
            self.save_checkpoint(val_loss, model, modelname, str)
            self.counter = 0
    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(),modelname+str+'.m')
        self.val_loss_min = val_loss