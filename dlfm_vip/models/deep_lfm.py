import torch
from math import ceil
from ..likelihoods import GaussianLikelihood
from ..layers import LFM
from ..variational.inducing_points import SVIStrategy
from ..utils import batch_assess


class DeepLFM(torch.nn.Module):
    def __init__(
        self,
        X,
        d_in,
        d_out,
        init_inducing_inputs,
        kernel='ode1',
        n_layers=2,
        d_inner=2,
        n_lf=2,
        n_basis_functions=16,
        mc=5,
        init_noise=1e-2,
        ts_mode=False,
        prior_cov_factor=1e-5,
        device="cpu",
        **kwargs,
    ):
        """
        PyTorch implementation of a deep latent force model with inference
        performed using a variational inducing points scheme and pathwise sampling.
        This is an OOTB implementation with a 1st order ODE kernel and Gaussian likelihood.

        :param X: Training inputs (N x d_in)
        :param d_in: Input dimensionality
        :param d_out: Output dimensionality
        :param init_inducing_inputs: Initial values for inducing input locations of first layer
        :param n_layers: Number of layers
        :param d_inner: Dimensionality of inner layers
        :param n_lf: No. latent forces per node
        :param n_basis_functions: Number of basis functions to use
        :param mc: Number of Monte Carlo samples
        :param init_noise: Initial value for likelihood noise variance
        :param ts_mode: Time series flag; if True, inducing locations fixed at first layer
        :param prior_cov_factor: Scaling factor for covariance of prior variational distribution
        :param device: Device to use for model, can be either 'cpu' or 'cuda'
        """
        super(DeepLFM, self).__init__()
        self.N_data = X.shape[0]
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.n_lf = n_lf
        self.n_basis_functions = n_basis_functions
        self.mc = mc
        self.ts_mode = ts_mode
        self.device = device
        self.likelihood = GaussianLikelihood(num_outputs=d_out, device=self.device, init_noise=init_noise)

        # Initialise model architecture, ensuring last layer has same number of outputs as target.
        # (currently assuming all layers of equal dimensionality, though this need not be the case)
        self.layers = torch.nn.ModuleList([])
        self.layer_dims = [d_in] + ((self.n_layers - 1) * [self.d_inner])
        self.num_outputs_per_layer = self.layer_dims[1:] + [self.d_out]

        assert self.layer_dims[0] == init_inducing_inputs.shape[-1]

        # By default, will initialise inner (i.e. non-output) layers to be nearly deterministic
        # (i.e. prior_cov_factor = 1e-5)
        init_cov_factors = [prior_cov_factor] * (self.n_layers - 1) + [1.0]

        # We now project the provided inputs into lower or higher dimensions in order to initialise the mean functions &
        # input process inducing locations in the internal layers; otherwise we could only consider the case in which
        # d_in == d_inner (i.e. input dimensionality equal to all layer dimensionalities)

        # If X is not small, initialise using a random subset of training input
        if X.shape[0] > 10000:
            rand_idx = torch.randint(
                X.shape[0], size=(10000,), dtype=torch.long, device=device
            )
            X = X[rand_idx, :]

        X_running = X.detach().clone()
        Z_running = init_inducing_inputs.detach().clone()

        for i in range(self.n_layers):
            d_in = self.layer_dims[i]
            d_out = self.num_outputs_per_layer[i]

            # If layer input and output dims match, no need to compute W
            if d_in == d_out:
                W = None

            # Initialise mean function using PCA projection if we need to step down
            elif d_in >= d_out:
                _, _, V = torch.linalg.svd(X_running, full_matrices=False)
                W = V[:d_out, :].T

            # Initialise using padding if we need to step up
            else:
                W = torch.cat(
                    [
                        torch.eye(
                            d_in,
                            requires_grad=False,
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.zeros(
                            (d_in, d_out - d_in),
                            requires_grad=False,
                            dtype=torch.float64,
                            device=device,
                        ),
                    ],
                    1,
                )

            self.layers.append(
                LFM(
                    self.N_data,
                    Z_running,
                    d_out,
                    kernel=kernel,
                    n_latent_forces=self.n_lf,
                    W=W,
                    n_basis_functions=self.n_basis_functions,
                    mc_samples=self.mc,
                    device=self.device,
                    prior_cov_factor_u=init_cov_factors[i],
                    **kwargs,
                )
            )

            if d_in != d_out:
                Z_running = Z_running.matmul(W)
                X_running = X_running.matmul(W)

        print(
            "\nInitial u-space lengthscales:",
            self.layers[0].u_gp.kernel.base_kernel.lengthscale,
            "\n",
        )

    def forward(self, x):
        """
        Propagate inputs forward through the model.
        """
        # Cast input to 3D as each slice [i,:,:] will be an MC realisation of the layer values
        x = x[None, :, :]
        model_input = x

        # Propagate input data through the architecture, feeding forward the inputs
        # at any layer besides the first or last of the network.
        output = x
        for i, layer in enumerate(self.layers):
            if (i == 0) or (i == len(self.layers) - 1):
                output = layer(output)
            else:
                output = layer(output, feed_forward_input=model_input)

        return output

    def forward_multiple_mc(self, x, S=100):
        """
        Allows for samples to be generated during evaluation
        with mc_multiplier times more MC samples than used during training.
        """
        if S < self.mc:
            mc_multiplier = 1
        else:
            mc_multiplier = int(ceil(S / self.mc))
        all_samps = None
        for i in range(mc_multiplier):
            with torch.no_grad():
                samps = self.forward(x)
                if all_samps is None:
                    all_samps = self.forward(x)
                else:
                    all_samps = torch.cat([all_samps, samps], 0)

        return all_samps[:S]

    def get_metrics(self, output_layer, y, y_pred_mean, y_scale=1.0, dim=None):
        "Compute MNLL, MSE, RMSE and normalised MSE."
        metrics = {}

        mnll = -torch.mean(
            torch.mean(
                self.likelihood.log_cond_prob(y, output_layer, y_scale=y_scale, dim=dim), 0
            )
        )
        nmse = torch.mean((y_pred_mean - y) ** 2) / torch.mean((torch.mean(y) - y) ** 2)
        mse = torch.mean(y_scale**2 * (y_pred_mean - y) ** 2)
        rmse = torch.sqrt(mse)

        metrics["mnll"] = mnll.item()
        metrics["nmse"] = nmse.item()
        metrics["mse"] = mse.item()
        metrics["rmse"] = rmse.item()
        return metrics

    def predict(self, x, y=None, y_scale=1.0, S=None):
        """
        Generate predictions for given input data. Note that when specifying
        S, it must be a multiple of self.mc to give the expected number of samples.
        """
        with torch.no_grad():
            # Sample self.mc samples
            if S is None:
                output_layer = self.forward(x)
            # Sample S samples
            else:
                output_layer = self.forward_multiple_mc(x, S=S)
            output = self.likelihood.predict(output_layer)
            y_pred_mean = torch.mean(output, 0)
            y_pred_std = torch.std(output, 0)

            # Compute MNLL, NMSE & NLPD if target values specified
            if y is not None:
                # If there are missing output observations, compute objective output by output
                if torch.isnan(y).any():
                    metrics_list = []
                    for i in range(self.d_out):
                        y_d = y[:, i]
                        y_pred_mean_d = y_pred_mean[:, i]
                        output_layer_d = output_layer[:, :, i]

                        # Remove missing (NaN) values from target and samples
                        is_nan = torch.isnan(y_d)
                        y_d = y_d[~is_nan]
                        output_layer_d = output_layer_d[:, ~is_nan]
                        y_pred_mean_d = y_pred_mean_d[~is_nan]
                        metrics_d = self.get_metrics(
                            output_layer_d, y_d, y_pred_mean_d, y_scale=y_scale, dim=i
                        )
                        metrics_list.append(metrics_d)

                        metrics = {}
                        # Average metrics across all outputs
                        for key in metrics_list[0].keys():
                            metrics[key] = sum(
                                [metrics_d[key] for metrics_d in metrics_list]
                            ) / len(metrics_list)

                        return y_pred_mean, y_pred_std, metrics

                # Otherwise, compute metrics all at once
                else:
                    metrics = self.get_metrics(
                        output_layer, y, y_pred_mean, y_scale=y_scale
                    )
                    return y_pred_mean, y_pred_std, metrics
            else:
                return y_pred_mean, y_pred_std

    def eval_step(
        self,
        data,
        data_valid,
        y_scale,
        current_iter,
        obj,
        batch_size,
    ):
        with torch.no_grad():
            X_train, y_train = data
            subset_size = min(self.N_data, batch_size)
            subset_idx = torch.randint(
                y_train.shape[0],
                size=(subset_size,),
                requires_grad=False,
                device=self.device,
            )
            train_metrics = batch_assess(
                self,
                X_train[subset_idx, :],
                y_train[subset_idx, :],
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
            metrics = batch_assess(
                self,
                *data_valid,
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
        print("Iteration %d" % (current_iter))
        print(
            "Bound = %.4f, Train NMSE = %.4f, Validation RMSE = %.4f, Validation NMSE = %.4f, Validation MNLL = %.4f\n"
            % (
                obj.item(),
                train_metrics["nmse"],
                metrics["rmse"],
                metrics["nmse"],
                metrics["mnll"],
            )
        )

        return metrics, train_metrics

    def train(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        lr=0.01,
        batch_size=64,
        n_iter=100,
        verbosity=1,
        ignore_nan=False,
        y_scale=1.0,
        model_filepath=None,
        fix_noise_iter=0,
    ):
        """
        :param X: Training inputs
        :param y: Training targets
        :param X_valid: Optional validation inputs
        :param y_valid: Optional validation targets
        :param lr: Learning rate
        :param batch_size: Batch size
        :param n_iter: Number of training iterations
        :param verbosity: Verbosity of metric evaluations
        :param ignore_nan: Must be set to True if observations in training data have NaN datapoints
        :param y_scale: Scaling parameter for outputs (used to compute evaluation metrics)
        :param model_filepath: File path for saving trained model (does not save if set to None)
        :param fix_noise_iter: Number of iterations to fix noise for at start of training
        """
        N = X.size()[0]  # Total number of training examples

        # Check that size of training set matches up with pre-supplied value used in computation of objective
        assert N == self.N_data

        # Initialise stochastic variational strategy for model inference
        self.svi = SVIStrategy(
            self.layers, self.likelihood, N, batch_size, ignore_nan=ignore_nan
        )

        # If no validation set specified, just evaluate on the training data
        if (X_valid is None) or (y_valid is None):
            X_valid = X.to(self.device)
            y_valid = y.to(self.device)

        else:
            X_valid = X_valid.to(self.device)
            y_valid = y_valid.to(self.device)

        # Set optimiser and parameters to be optimised
        pars = dict(self.named_parameters())
        for p in list(pars):
            if ("noise") in p:
                pars.pop(p, None)
                fitting_noise = False
            # Fix initial layer IPs for time series problems
            if ("layers.0.u_gp.inducing_inputs") in p:
                if self.ts_mode:
                    pars.pop(p, None)
        opt = torch.optim.AdamW(pars.values(), lr=lr)

        # Initialise dataloader for minibatch training
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Perform iterations of minibatch training
        current_iter = 0
        training = True
        while training:
            # Use each batch to train model
            for X_minibatch, y_minibatch in train_dataloader:
                opt.zero_grad()
                output_layer = self.forward(X_minibatch.to(self.device))
                obj = self.svi.nelbo(output_layer, y_minibatch.to(self.device))
                obj.backward()
                opt.step()
                current_iter += 1

                # Stop training if specified number of iterations has been completed
                if current_iter >= n_iter:
                    print("\nTraining Complete.\n")
                    training = False
                    break

                # Un-fix noise after specified number of training iterations completed
                if (not fitting_noise) and (current_iter > fix_noise_iter):
                    opt.add_param_group({"params": self.likelihood.log_noise})
                    print("\nNow fitting noise...\n")
                    fitting_noise = True

                # Display validation metrics at specified intervals
                if verbosity == 0:
                    print("Iteration %d complete." % current_iter)
                elif current_iter % verbosity == 0:
                    _, _ = self.eval_step(
                        (
                            X_minibatch.to(self.device),
                            y_minibatch.to(self.device),
                        ),  # NOTE - using this instead of `data` to save memory
                        (X_valid, y_valid),
                        y_scale,
                        current_iter,
                        obj,
                        batch_size,
                    )

        # Save the final model once training is complete, if applicable
        if model_filepath is not None:
            torch.save(self.state_dict(), model_filepath)
