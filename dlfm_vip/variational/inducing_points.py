import torch


class SVIStrategy(torch.nn.Module):
    """
    Implementation of stochastic variational inference for the interdomain DLFM;
    takes ModuleList of layers as input.
    """

    def __init__(self, layers, likelihood, N, batch_size, ignore_nan=False):
        super(SVIStrategy, self).__init__()
        self.layers = layers
        self.likelihood = likelihood
        self.N = N
        self.batch_size = batch_size
        self.ignore_nan = ignore_nan

    def ell(self, output, y):
        """
        Computes the expected log likelihood term of the ELBO.
        Args:
            output: output from the model    
            y: target observations 
        """
        # If there are missing output observations, compute objective output by output
        if torch.isnan(y).any():
            ell = 0.0
            for i in range(self.likelihood.num_outputs):
                y_d = y[:, i]
                output_d = output[:, :, i]

                # Remove missing (NaN) values from target and samples
                is_nan = torch.isnan(y_d)
                y_d = y_d[~is_nan]
                output_d = output_d[:, ~is_nan]

                ell += torch.sum(
                    torch.mean(self.likelihood.log_cond_prob(y_d, output_d), 0)
                ) * (self.N / output.shape[1])

        # Otherwise, compute likelihood all at once
        else:
            ell = torch.sum(torch.mean(self.likelihood.log_cond_prob(y, output), 0)) * (
                self.N / output.shape[1]
            )

        return ell

    def get_model_kl(self):
        """Compute KL divergence term of ELBO."""
        kl = 0
        for layer in self.layers:
            kl += layer.compute_KL()
        return kl

    def nelbo(self, output, y):
        """
        Get negative ELBO; minimising NELBO == maximising ELBO.
        """
        kl = self.get_model_kl()
        ell = self.ell(output, y)
        return kl - ell
