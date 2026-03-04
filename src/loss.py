import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTMagnitudeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Pérdida espectral multi-objetivo: convergencia espectral (alpha),
        magnitud logarítmica L1 (beta) y MSE complejo (gamma).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_hat, y):
        # y_hat, y: (B, 2, F, T) — canal 0: real, canal 1: imaginario

        # Magnitud espectral con epsilon para estabilidad numérica
        mag_hat = torch.sqrt(y_hat[:, 0, :, :]**2 + y_hat[:, 1, :, :]**2 + 1e-7)
        mag_y = torch.sqrt(y[:, 0, :, :]**2 + y[:, 1, :, :]**2 + 1e-7)

        # 1. Spectral Convergence: error relativo en norma Frobenius (escala lineal)
        sc_loss = torch.norm(mag_y - mag_hat, p="fro") / (torch.norm(mag_y, p="fro") + 1e-7)

        # 2. Log-Magnitude L1: error perceptual en escala logarítmica
        log_mag_hat = torch.log(mag_hat)
        log_mag_y = torch.log(mag_y)
        log_mag_loss = F.l1_loss(log_mag_hat, log_mag_y)

        # 3. Complex MSE: supervisión directa de fase y magnitud sobre (real, imag)
        complex_mse = F.mse_loss(y_hat, y)

        return (self.alpha * sc_loss) + (self.beta * log_mag_loss) + (self.gamma * complex_mse)