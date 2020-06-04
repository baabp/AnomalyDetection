from skorch import NeuralNetRegressor
import torch


class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + loss_l1