import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTMagnitudeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, freq_emphasis=True):
        """
        Pérdida espectral multi-objetivo: convergencia espectral (alpha),
        magnitud logarítmica L1 (beta) y MSE complejo (gamma).

        freq_emphasis: si True, pondera las altas frecuencias con más peso,
                       ya que son las que la red debe reconstruir en super-resolución.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.freq_emphasis = freq_emphasis

    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Magnitud espectral a partir de (B, 2, F, T) → (B, F, T)."""
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-7)

    def _freq_weights(self, n_freqs: int, device: torch.device) -> torch.Tensor:
        """
        Máscara de énfasis frecuencial: escala lineal de 1.0 (DC) a 2.0 (Nyquist).
        Shape: (1, F, 1) — broadcast sobre batch y tiempo.
        Las altas frecuencias pesan el doble que las bajas.
        """
        weights = torch.linspace(1.0, 2.0, n_freqs, device=device)
        return weights.view(1, -1, 1)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_hat, y: (B, 2, F, T) — canal 0: real, canal 1: imaginario

        mag_hat = self._magnitude(y_hat)   # (B, F, T)
        mag_y   = self._magnitude(y)       # (B, F, T)

        # Énfasis en altas frecuencias
        if self.freq_emphasis:
            w = self._freq_weights(mag_y.size(1), mag_y.device)
            mag_hat = mag_hat * w
            mag_y   = mag_y   * w

        # 1. Spectral Convergence — calculada por muestra y luego promediada
        sc_loss = (torch.norm(mag_y - mag_hat, p="fro", dim=(-2, -1)) /(torch.norm(mag_y, p="fro", dim=(-2, -1)) + 1e-7)).mean()

        # 2. Log-Magnitude L1 — error perceptual en escala logarítmica
        log_loss = F.l1_loss(torch.log(mag_hat), torch.log(mag_y))

        # 3. Complex MSE — supervisión directa de fase y magnitud sobre (real, imag)
        complex_mse = F.mse_loss(y_hat, y)

        return (self.alpha * sc_loss) + (self.beta * log_loss) + (self.gamma * complex_mse)