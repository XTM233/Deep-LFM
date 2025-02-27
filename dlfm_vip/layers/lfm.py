import torch
import gpytorch
from math import pi, log
from ..kernels import ODE1Kernel

torch.set_default_dtype(torch.float64)


class LFM(torch.nn.Module):
    """
    Implements a multi-output LFM with inducing points and pathwise sampling, with IPs shared
    across the input dimensions. Similar to the NP-CGP of McDonald et al. (2022), except with no
    GP over the filter (instead, the filter/Green's function 'G' has a fixed functional form) and
    no interdomain transform for the inducing points.
    """

    def __init__(
        self,
        N_data,
        init_u_inducing_inputs,
        num_outputs,
        kernel="ode1",
        n_latent_forces=None,
        W=None,
        init_u_lengthscale=0.3,
        n_basis_functions=16,
        mc_samples=1,
        device="cpu",
        init_amp=1.0,
        prior_cov_factor_u=1.0,
        gamma_init=0.01,
        alpha_init=0.1,
        omega_init=2.0,
        **kwargs,
    ):
        super(LFM, self).__init__()
        self.N_data = N_data
        self.d_out = num_outputs
        self.d_in = init_u_inducing_inputs.shape[1]
        self.W = W
        self.num_u_inducing_points = init_u_inducing_inputs.shape[0]
        self.init_u_inducing_inputs = init_u_inducing_inputs
        self.init_u_lengthscale = init_u_lengthscale
        if n_latent_forces is None:
            n_latent_forces = self.d_out
        self.n_latent_forces = n_latent_forces
        self.n_basis_functions = n_basis_functions
        self.mc_samples = mc_samples
        self.device = device

        if kernel == "ode1":
            self.kernel = ODE1Kernel(
                self.d_in,
                self.d_out,
                gamma_init=gamma_init,
                device=self.device,
            )
        elif kernel == "ode2":
            self.kernel = ODE2Kernel(
                self.d_in,
                self.d_out,
                omega_init=omega_init,
                alpha_init=alpha_init,
                device=self.device,
            )
        elif kernel == "composite":
            self.kernel = CompositeKernel(
                self.d_in,
                self.d_out,
                gamma_init=gamma_init,
                omega_init=omega_init,
                alpha_init=alpha_init,
                device=self.device,
            )
        else:
            raise NotImplementedError(
                "Enter a valid kernel from: ode1, ode2, composite."
            )

        self.u_gp = PathwiseGP(
            self.init_u_inducing_inputs,
            mc_samples=self.mc_samples,
            num_basis_functions=self.n_basis_functions,
            init_lengthscale=self.init_u_lengthscale,
            device=self.device,
            prior_cov_factor=prior_cov_factor_u,
            num_gps=self.n_latent_forces,
            **kwargs,
        )

        # Initialise amplitudes (i.e. a_{d, q}, as in NP-CGP)
        self.register_parameter(
            "log_amps",
            torch.nn.Parameter(
                torch.ones(
                    self.d_out, self.n_latent_forces, dtype=torch.float64, device=device
                )
                * log(init_amp)
            ),
        )

    @property
    def amps(self):
        return torch.exp(self.log_amps)

    def __call__(self, xs, feed_forward_input=None):
        '''
        Args:
            xs: N x D tensor of inputs
            feed_forward_input: N x D tensor of inputs to the final layer
        Returns:
            layer_out: S x N x D tensor of outputs, where
                S: Number of Monte Carlo samples.
                N: Number of data points.
                D: Output dimensionality.
        '''
        u_basis = self.u_gp.sample_basis()
        thetaus, betaus, wus = u_basis
        qus = self.u_gp.compute_q(u_basis)
        zus = self.u_gp.inducing_inputs
        thetaus = torch.swapaxes(thetaus, 2, 3)
        u_ls = self.u_gp.kernel.base_kernel.lengthscale
        u_var = self.u_gp.kernel.outputscale
        out = []

        for i in range(self.d_out):
            outi = self.kernel.convolution_integral(
                i,
                xs,
                wus,
                u_ls,
                u_var,
                thetaus,
                betaus,
                zus,
                qus,
            )
            out.append(outi)

        layer_out = torch.stack(out, -1)  # S x N x Q x D
        layer_out = (self.amps.T[None, None, :, :] * layer_out).sum(axis=2)  # S x N x D

        # Apply mean function if not at the final layer
        if feed_forward_input is not None:
            if self.W is None:
                layer_out = layer_out + feed_forward_input
            else:
                layer_out = layer_out + feed_forward_input.matmul(self.W)

        return layer_out

    def compute_KL(self):
        return self.u_gp.compute_KL()


class PathwiseGP(torch.nn.Module):
    """
    Base approx. multi-input/output GP class which efficiently samples as
    per Wilson et al. [2020]. Used for the input process in our model.
    (Standard GP kernel with optimised inter-domain IPs and whitening)
    """

    def __init__(
        self,
        init_inducing_inputs,
        inducing_outputs=None,
        mc_samples=11,
        init_lengthscale=0.1,
        num_basis_functions=50,
        prior_cov_factor=1.0,
        jitter=1e-7,
        device="cpu",
        whiten=True,
        num_gps=1,
    ):
        super(PathwiseGP, self).__init__()
        self.d_in = init_inducing_inputs.shape[1]
        self.num_inducing_points = init_inducing_inputs.shape[0]
        self.num_basis_functions = num_basis_functions
        self.device = device
        self.mc_samples = mc_samples
        self.whiten = whiten
        self.jitter = jitter
        self.num_gps = num_gps
        self.inducing_outputs = inducing_outputs

        self.register_parameter(
            "inducing_inputs", torch.nn.Parameter(init_inducing_inputs)
        )

        # Initialise kernel and lengthscales
        self.kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.d_in, device=device)
        )
        self.kernel.base_kernel.lengthscale = torch.tensor(
            [[init_lengthscale] * self.d_in], device=device, dtype=torch.float64
        )
        self.kernel.to(device)

        # Initialise variational distribution
        batch_shape = torch.Size([num_gps])
        self.variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing_points, device=device, batch_shape=batch_shape
        )

        # Initialise inducing function covariance to identity, scaled by prior_cov_factor
        # (should typically be small for inner layers, but ~1.0 at output layer)
        init_variational_covar = (
            torch.eye(self.num_inducing_points, dtype=torch.float64, device=device)[
                None, :, :
            ].repeat(*batch_shape, 1, 1)
            * prior_cov_factor
        )
        self.variational_dist.chol_variational_covar = torch.nn.Parameter(
            init_variational_covar
        )
        self.variational_dist.to(device)

    @property
    def prior(self):
        mean = torch.zeros(
            self.num_inducing_points, dtype=torch.float64, device=self.device
        )
        cov = self.kernel.forward(
            self.inducing_inputs, self.inducing_inputs
        ) + self.jitter * torch.eye(
            self.num_inducing_points,
            requires_grad=False,
            dtype=torch.float64,
            device=self.device,
        )
        return gpytorch.distributions.MultivariateNormal(mean, cov, validate_args=True)

    def sample_basis(self):
        # thets has shape (S, Q, P, B, N)
        thets = (
            torch.randn(
                self.mc_samples,
                self.num_gps,
                self.d_in,
                self.num_basis_functions,
                requires_grad=False,
                dtype=torch.float64,
                device=self.device,
            )
            / self.kernel.base_kernel.lengthscale.squeeze(0)[None, None, :, None]
        )
        ws = torch.sqrt(
            torch.tensor(
                2.0 / self.num_basis_functions,
                requires_grad=False,
                dtype=torch.float64,
                device=self.device,
            )
        ) * torch.randn(
            self.mc_samples,
            self.num_gps,
            self.num_basis_functions,
            requires_grad=False,
            dtype=torch.float64,
            device=self.device,
        )
        betas = (
            2
            * pi
            * torch.rand(
                self.mc_samples,
                self.num_gps,
                self.num_basis_functions,
                requires_grad=False,
                dtype=torch.float64,
                device=self.device,
            )
        )
        return thets, betas, ws

    def compute_q(self, basis):
        # NOTE what are the qus?
        thets, betas, ws = basis
        # this is the same as example basis function in Efficient Sampling Wilson2020, beta equivalent to tau
        phiz = torch.cos(self.inducing_inputs.matmul(thets) + betas[:, :, None, :])
        LKzz = torch.linalg.cholesky(self.prior.covariance_matrix)

        if self.inducing_outputs is None:
            us = self.variational_dist.forward().rsample(
                sample_shape=torch.Size([self.mc_samples])
            )
        else:
            us = self.inducing_outputs[None, None, :]

        # Whitening, as introduced in Sec 2.1 of Murray et al. (2010)
        if self.whiten:
            us = us.matmul(LKzz)

        x = us[:, :, :, None] - phiz.matmul(ws[:, :, :, None])

        return torch.cholesky_solve(x, LKzz[None, None, :, :]).squeeze(-1)

    def __call__(self, x):
        if len(x.shape) == 2:
            x = x[None, :, :]
        basis = self.sample_basis()
        thets, betas, ws = basis
        phix = torch.cos(
            torch.einsum("bxd, bqdn ->bqxn", x, thets) + betas[:, :, None, :]
        )
        qs = self.compute_q(basis)
        Kxz = self.kernel.forward(x, self.inducing_inputs)
        basis_part = torch.einsum("bqz, bxz -> bqx", qs, Kxz) # update, qs = v in Wilson2020
        random_part = torch.einsum("bqn, bqxn -> bqx", ws, phix) # prior
        return basis_part + random_part

    def compute_KL(self):
        kl = sum(
            [
                torch.distributions.kl.kl_divergence(disti, self.prior)
                for disti in self.variational_dist.forward()
            ]
        )
        return kl
