import torch
import numpy as np
from math import sqrt, pi


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def ls2pr(ls):
    return 1 / (2 * ls**2)


def pr2ls(pr):
    return (1 / (2 * pr)) ** 0.5


def erfc_complex(z, n=3):
    """Approximate computation of erfc, required as unsupported by PyTorch for 
       complex inputs. Whilst our approximation of erfc(z) works just fine for the
       1st and 4th quadrants of the complex plane, we must instead use 
       erfc(z) = 2 - erfc(-z) for complex numbers in the 2nd and 3rd quadrants."""

    def erfc_approx(z):
        """Approximation to complementary error function, required for ODE2 kernel
            computation as PyTorch does not support complex arguments for erfc().

            See https://en.wikipedia.org/wiki/Error_function#Factorial_series"""
        coeff = torch.exp(-(z**2)) / (z * sqrt(pi))
        z2 = z**2
        n1 = 1
        n2 = 1 / (2 * (z2 + 1))
        if n == 2:
            return coeff * (n1 - n2)
        elif n == 3:
            n3 = 1 / (4 * (z2 + 1) * (z2 + 2))
            return coeff * (n1 - n2 + n3)
        else:
            raise NotImplementedError("Enter n=2 or n=3 for erfc approximation.")

    # Handle the aforementioned quadrant issue
    out = torch.zeros_like(z)
    out[(z.real > 0) & (z.imag > 0)] = erfc_approx(z[(z.real > 0) & (z.imag > 0)])
    out[(z.real > 0) & (z.imag < 0)] = erfc_approx(z[(z.real > 0) & (z.imag < 0)])
    out[(z.real < 0) & (z.imag > 0)] = 2 - erfc_approx(- z[(z.real < 0) & (z.imag > 0)])
    out[(z.real < 0) & (z.imag < 0)] = 2 - erfc_approx(- z[(z.real < 0) & (z.imag < 0)])
    
    return out

def dkl_gaussian(m_q, lv_q, m_p, lv_p):
    """Returns the Kullback Leibler divergence for MV Gaussian distributions
        q and p with diagonal covariance matrices.

    :param m_q: Means for q
    :param lv_q: Log-variances for q
    :param m_p: Means for p
    :param lv_p: Log-variances for p
    :return: KL(q||p)
    """

    # Compute constituent terms of DKL
    term_a = lv_p - lv_q
    term_b = torch.pow(m_q - m_p, 2) / torch.exp(lv_p)
    term_c = torch.exp(lv_q - lv_p) - 1

    return 0.5 * torch.sum(term_a + term_b + term_c)


def batch_assess(
    model, X, Y, y_scale=None, device="cpu", batch_size=1000, S=100, task="regression", output_wise=False
):
    """
    Function to assess metrics in batches, adapted from:
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    """
    if y_scale is None:
        y_scale = 1.0

    n_batches = max(int(X.shape[0] / batch_size), 1)
    temp_all_metrics = {"mnll": [], "mse": [], "nmse": []}
    all_metrics = {"mnll": [], "mse": [], "nmse": []}

    for X_batch, Y_batch in zip(
        np.array_split(X, n_batches), np.array_split(Y, n_batches)
    ):

        if output_wise:
            _, _, metrics = model.predict_outputs(
                X_batch.to(device),
                y=Y_batch.to(device),
                y_scale=y_scale,
                S=S,
            )
        else:
            _, _, metrics = model.predict(
                X_batch.to(device),
                y=Y_batch.to(device),
                y_scale=y_scale,
                S=S,
            )

        temp_all_metrics["mnll"].append(metrics["mnll"])
        temp_all_metrics["mse"].append(metrics["mse"])
        temp_all_metrics["nmse"].append(metrics["nmse"])

    if output_wise:
        for d in range(Y_batch.shape[1]):
            all_metrics["nmse"].append(np.mean(np.array([batch[d] for batch in temp_all_metrics["nmse"]])))
            all_metrics["mse"].append(np.mean(np.array([batch[d] for batch in temp_all_metrics["mse"]])))
            all_metrics["rmse"] = np.sqrt(all_metrics["mse"])
            all_metrics["mnll"].append(np.mean(np.array([batch[d] for batch in temp_all_metrics["mnll"]])))
    else:
        all_metrics["nmse"] = np.mean(np.array(temp_all_metrics["nmse"]))
        all_metrics["mse"] = np.mean(np.array(temp_all_metrics["mse"]))
        all_metrics["rmse"] = np.sqrt(temp_all_metrics["mse"])[0]
        all_metrics["mnll"] = np.mean(np.array(temp_all_metrics["mnll"]))

    return all_metrics
