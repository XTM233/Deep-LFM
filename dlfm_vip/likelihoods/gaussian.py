import torch
import numpy as np
from .likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    def __init__(self, num_outputs=1, init_noise=0.01, device="cpu"):
        super(GaussianLikelihood, self).__init__()
        self.device = device
        self.num_outputs = num_outputs

        # Log noise/variance parameter of output Gaussian likelihood
        self.register_parameter(
            "log_noise",
            torch.nn.Parameter(
                torch.log(
                    torch.tensor(
                        [init_noise] * self.num_outputs,
                        dtype=torch.float64,
                        requires_grad=True,
                        device=self.device,
                    )
                )
            ),
        )

    def log_cond_prob(self, output, latent, y_scale=1.0, dim=None):
        if type(y_scale) is not torch.TensorType:
            y_scale = torch.tensor(
                [y_scale], dtype=torch.float64, device=self.device, requires_grad=False
            )
        if dim is None:
        # NOTE what's y_scale for?
        # NOTE latent or target? is there a ground truth? 
        # when called, the inputs are y_d, output_d
            log_cond_prob = (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * torch.log(y_scale**2 * torch.exp(self.log_noise))
                - (0.5 * torch.square(y_scale * output - y_scale * latent))[:, :, None]
                / (torch.exp(self.log_noise) * y_scale**2)
            )
        else:
            # This case catches the scenario where we have missing values (NaNs) to impute and thus
            # are forced to compute the ELBO output-by-output as opposed to all at once
            log_cond_prob = (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * torch.log(y_scale**2 * torch.exp(self.log_noise[dim]))
                - (0.5 * torch.square(y_scale * output - y_scale * latent))
                / (torch.exp(self.log_noise[dim]) * y_scale**2)
            )

        return log_cond_prob

    def predict(self, latent):
        return latent
