import torch
from math import sqrt, pi


class ODE1Kernel(torch.nn.Module):
    """
    1st order ODE LFM kernel for use with the standard variational LFM/DLFM classes.
    """

    def __init__(
        self,
        d_in,
        d_out,
        gamma_init=0.01,
        device="cpu",
    ):
        super(ODE1Kernel, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device

        # Initialise gamma (i.e. decay) ODE1 kernel hyperparameters
        self.register_parameter(
            "log_gamma",
            torch.nn.Parameter(
                torch.log(
                    torch.ones(
                        self.d_in, self.d_out, dtype=torch.float64, device=self.device
                    )
                    * gamma_init
                )
            ),
        ),

    def G(self, x, gamma):
        """Green's function corresponding to ODE1 kernel; pointwise operation designed
        for use in computing G_d^{(p)} in expressions from paper."""
        return torch.exp(-gamma * x)

    def convolution_integral(
        self,
        d,  # output to sample from (d=1,...D)
        xs,  # N x P or S x N x P
        wus,  # S x Q x B
        u_ls,  # P
        u_var,  # Scalar
        thetaus,  # S x Q x B x P
        betaus,  # S x Q x B
        zus,  # M x P
        qus,  # S x Q x M
    ):
        """Function which computes the convolution integral of the
        ODE1 Green's function with a latent GP."""
        # NOTE this convolution does not use RFF
        # Get variables and convert to align with shape S x N x Q x B x P...
        gamma = torch.exp(self.log_gamma[:, d])[None, None, None, None, :]
        B = wus.shape[-1]
        wus = wus[:, None, :, :, None]
        thetaus = thetaus[:, None, :, :, :]
        xs = xs[:, :, None, None, :]
        betaus = betaus[:, None, :, :, None]
        # ... or S x N x Q x M x P, as required.
        qus = qus[:, None, :, :, None]
        zus = zus[None, None, None, :, :]
        u_ls = u_ls.squeeze(0)[None, None, None, None, :]

        # Define 0 + 1*i as a variable for convenience whilst computing integral
        imag_1 = torch.zeros(1, dtype=torch.cfloat, device=self.device)  # i.e. 0 + 1i
        imag_1.imag = torch.tensor([1.0], dtype=torch.float64, device=self.device)

        # Compute first part of integral, which is summed over B basis functions
        # (see Section 1.1 of supplemental material for details)
        I1 = (
            wus.squeeze(-1)
            * (
                torch.prod(
                    (torch.exp(imag_1 * (thetaus * xs + betaus)))
                    / (gamma + imag_1 * thetaus),
                    -1,
                )
            ).real
        )  # S x N x Q x B
        I1 = sqrt(2.0 / B) * I1.sum(3)  # S x N x Q

        # Compute second part of integral, which is summed over M inducing points
        I2 = (
            u_ls
            * sqrt(pi / 2)
            * torch.exp((gamma / 2.0) * (gamma * u_ls**2 + 2 * zus - 2 * xs))
        ) * (
            torch.erfc((gamma * u_ls**2 + zus - xs) / (u_ls * sqrt(2)))
        )  # S x N x Q x M x P
        I2 = I2.prod(-1) * qus.squeeze(-1)
        I2 = I2.sum(3) * u_var  # S x N x Q

        # Combine terms
        return I1.real + I2  # S x N x Q
