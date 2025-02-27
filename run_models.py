import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)
import torch
import random
import matplotlib
import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import integrate
from dlfm_vip.models import DeepLFM # TODO change accordingly for different models
from dlfm_vip.utils import batch_assess

torch.set_default_dtype(torch.float64)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add to remove type 3 fonts for PDF
plt.rcParams["pdf.fonttype"] = 42


def scale_data(X_tr, y_tr, X_test_extrap, y_test_extrap, test_intra=False, **kwargs):
    """
    Scale inputs & outputs, after splitting into train and test sets
    Each of the arguments is a numpy array of shape (T,)
    Returns the scaled data in (T, D) and the scalers used
    """
    ss_x = MinMaxScaler()
    X_tr = ss_x.fit_transform(X_tr.reshape(-1, 1))
    X_test_extrap = ss_x.transform(X_test_extrap.reshape(-1, 1))

    ss_y = StandardScaler()
    y_tr = ss_y.fit_transform(y_tr.reshape(-1, 1))
    y_test_extrap = ss_y.transform(y_test_extrap.reshape(-1, 1))
    
    if test_intra:
        X_test_interp = ss_x.transform(kwargs["X_test_interp"].reshape(-1, 1))
        y_test_interp = ss_y.transform(kwargs["y_test_interp"].reshape(-1, 1))
        return ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap, X_test_interp, y_test_interp

    return ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap


def run(args):
    # CUDA initialisations (Note, if you aren't using a GPU, you'll likely
    # need to train for longer than specified in this demo)
    SEED = args.seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    cuda_ = f"cuda:{int(args.cuda)}" if args.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(cuda_)
    print("Device:", device)

    str_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, args.dataset, str_time)
    if args.dataset == "toy":
        # TODO save and load this dataset
        # Generate toy data from a hierarchical ODE system
        t = np.linspace(0, 15, 750)
        f1_init, f2_init = [0], [0]
        decay1, decay2 = [0.01], [0.02]
        tau = np.zeros(750)
        lambd = 1

        def u_func(t, array=None):
            return np.cos(t / 2) + 6 * np.sin(3 * t)

        def f1(tau):
            i = np.array([0 + 1j])
            G1 = (np.exp(i * lambd * (t - tau)) - np.exp(-decay1[0] * (t - tau))) / (
                decay1[0] + i * lambd
            )
            return G1 * u_func(tau)

        f1 = integrate.quad_vec(f1, 0, max(t))[0]

        def f2(tau):
            i = np.array([0 + 1j])
            G2 = (np.exp(i * lambd * (f1 - tau)) - np.exp(-decay2[0] * (f1 - tau))) / (
                decay2[0] + i * lambd
            )
            return G2 * u_func(tau)

        f2 = integrate.quad_vec(f2, 0, max(f1))[0].real
        f2 += 0.04 * np.random.randn(*f2.shape)  # Add Gaussian noise to outputs

        if args.test_intra:
            X_tr = np.concatenate([t[:125], t[225:600]])
            y_tr = np.concatenate([f2[:125], f2[225:600]])
            ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap, X_test_interp, y_test_interp = scale_data(
                X_tr, y_tr, t[600:], f2[600:], test_intra=args.test_intra, 
                X_test_interp=t[125:225], y_test_interp=f2[125:225]
                )
        else:
            ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap = scale_data(
                t[:600], f2[:600], t[600:], f2[600:])
    
    elif "charis" in args.dataset:
        # eg dataset = charis_abp, case insensitive
        _, col = args.dataset.split("_")
        # TODO write splitting point into args?
        t = np.linspace(0, 1, 1000)
        df = pd.read_csv("bin/data/charis6.csv")
        f2 = df[col.upper()].values # NOTE - assuming the column name is the same as the dataset name
        if args.test_intra:
            if col == "abp":
                i1, i2, i3 = 200, 350, 975
            elif col == "ecg":
                i1, i2, i3 = 400, 550, 975
            elif col == "icp":
                i1, i2, i3 = 600, 750, 975
            X_tr = np.concatenate([t[:i1], t[i2:i3]])
            y_tr = np.concatenate([f2[:i1], f2[i2:i3]])
            ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap, X_test_interp, y_test_interp = scale_data(
                X_tr, y_tr, t[i3:], f2[i3:], test_intra=args.test_intra, 
                X_test_interp=t[i1:i2], y_test_interp=f2[i1:i2]
                )
        else:
            ss_x, ss_y, X_tr, y_tr, X_test_extrap, y_test_extrap = scale_data(
                t[:975], f2[:975], t[975:], f2[975:])
    # Scale inputs & outputs, and split into train and test sets

    # ss_x = MinMaxScaler()
    # X = t.reshape(-1, 1)
    # if args.test_intra:
    #     X_tr = ss_x.fit_transform(np.concatenate([t[:125], t[225:600]]).reshape(-1, 1))
    #     X_test_interp = ss_x.transform(t[125:225].reshape(-1, 1))
    #     X_test_extrap = ss_x.transform(t[600:].reshape(-1, 1))
    # else:
    #     X_tr = ss_x.fit_transform(t[:600].reshape(-1, 1))
    #     X_test_extrap = ss_x.transform(t[600:].reshape(-1, 1))

    # ss_y = StandardScaler()
    # if args.test_intra:
    #     y_tr = ss_y.fit_transform(
    #         np.concatenate([f2[0, :125], f2[0, 225:600]]).reshape(-1, 1)
    #     )
    #     y_test_interp = ss_y.transform(f2[0, 125:225].reshape(-1, 1))
    #     y_test_extrap = ss_y.transform(f2[0, 600:].reshape(-1, 1))
    # else:
    #     y_tr = ss_y.fit_transform(f2[0, :600].reshape(-1, 1))
    #     y_test_extrap = ss_y.transform(f2[0, 600:].reshape(-1, 1))

    y_scale = torch.tensor([ss_y.scale_], dtype=torch.float64, device=device)
    print(y_scale)
    if not args.saved_model:
        
        # Set model and training arguments
        main_kwargs = {
            "n_basis_functions": args.n_basis,
            "mc": args.mc,
            "init_noise": 1e-2,
            "n_layers": 1 if args.dataset == "toy" else 2,
            "n_lf": 1, # NOTE number of latent forces?
            "d_inner": 3,
            "init_amp": 1.0,
            "init_u_lengthscale": 0.1,
            "jitter": 1e-3,
            "gamma_init": 2.5 if "charis" in args.dataset else 1.0, # initial decay parameter
            "ts_mode": True,
            "prior_cov_factor": 1.0,  # NOTE - using 1.0 for TS data
            "device": str(device),
        }

        train_kwargs = {
            "lr": 1e-2,
            "n_iter": args.n_iter,
            "verbosity": args.verbosity,
            "batch_size": args.batch_size,
            "fix_noise_iter": 10000,
        }

        # Initialise evenly spaced IPs for first layer as this is a time series problem
        X = t.reshape(-1, 1)
        init_inducing_inputs = np.linspace(
            ss_x.transform(X).min(),
            ss_x.transform(X).max(),
            args.num_ips,
        ).reshape(-1, 1)

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Initialise model
        dlfm = DeepLFM(
            torch.tensor(X_tr, dtype=torch.float64, device=device, requires_grad=False),
            d_in=1,
            d_out=1,
            init_inducing_inputs=torch.tensor(
                init_inducing_inputs, dtype=torch.float64, device=device
            ),
            **main_kwargs,
        ).to(device)

        try:
            # Train model
            dlfm.train(
                torch.tensor(X_tr, dtype=torch.float64, device=device, requires_grad=False),
                torch.tensor(y_tr, dtype=torch.float64, device=device, requires_grad=False),
                torch.tensor(
                    X_test_extrap, dtype=torch.float64, device=device, requires_grad=False
                ),
                torch.tensor(
                    y_test_extrap, dtype=torch.float64, device=device, requires_grad=False
                ),
                y_scale=y_scale,
                ignore_nan=False,
                model_filepath=os.path.join(save_dir, "model.torch"),
                **train_kwargs,
            )
        except KeyboardInterrupt:
            print("Training interrupted, saving model")
            torch.save(dlfm.state_dict(), os.path.join(save_dir, "model.torch"))
        all_args = {**train_kwargs, **main_kwargs, "n_ips": args.num_ips, "seed": SEED, "dataset": args.dataset}

        # Write all arguments to file
        with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(all_args, f, ensure_ascii=False, indent=4)
    else:
        save_dir = args.saved_model

    # Load saved model, compute final test set predictions and output to file
    dlfm.load_state_dict(torch.load(os.path.join(save_dir, "model.torch")))

    res_dict = batch_assess(
        dlfm,
        torch.tensor(
            X_test_extrap, dtype=torch.float64, device=device, requires_grad=False
        ),
        torch.tensor(
            y_test_extrap, dtype=torch.float64, device=device, requires_grad=False
        ),
        y_scale=y_scale,
        device=device,
        S=500,
        output_wise=False
    )

    print(res_dict)

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    # Plot model results and save
    with torch.no_grad():
        # Compute predictions across all data points
        X_all = ss_x.transform(t.reshape(-1, 1))
        mean_all, std_all = dlfm.predict(
            torch.tensor(X_all, dtype=torch.float64, device=device)
        )
        mean_all, std_all = (
            mean_all.detach().cpu().numpy().flatten(),
            std_all.detach().cpu().numpy().flatten(),
        )

        # Plot predictions for model
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.scatter(
            X_tr.flatten(),
            y_tr.flatten(),
            label="Training Data",
            color="grey",
            s=1,
            alpha=0.5,
        )
        if args.test_intra:
            ax.scatter(
                X_test_interp.flatten(),
                y_test_interp.flatten(),
                label="Test Data (Interpolation)",
                s=1,
                color="orange",
                alpha=0.5,
            )
            ax.scatter(
                X_test_extrap.flatten(),
                y_test_extrap.flatten(),
                label="Test Data (Extrapolation)",
                s=1,
                color="red",
                alpha=0.5,
            )
        else:
            ax.scatter(
                X_test_extrap.flatten(),
                y_test_extrap.flatten(),
                label="Test Data",
                s=1,
                color="red",
                alpha=0.5,
            )
        ax.plot(
            X_all.flatten(),
            mean_all.flatten(),
            label="Predictive Mean (Train)",
            color="purple",
        )
        ax.fill_between(
            X_all.flatten(),
            mean_all.flatten() + 2.0 * std_all.flatten(),
            mean_all.flatten() - 2.0 * std_all.flatten(),
            alpha=0.2,
            color="black",
            linewidth=0.1,
        )
        ax.set_xlabel("$t$", fontsize=16)
        ax.set_ylabel("$y_2$", fontsize=16)
        ax.set_ylim(bottom=-6.0, top=3.75)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "toy.pdf"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy experiment.")
    parser.add_argument("--cuda", default=None, type=int, help="Specify CUDA device, e.g., '0' for cuda:0. If not specified, use CPU.")
    parser.add_argument("--seed", default=1, type=int) # 1,2,3,4,5
    parser.add_argument("--mc", default=50, type=int)
    parser.add_argument("--n_basis", default=16, type=int)
    parser.add_argument("--num_ips", default=20, type=int)
    parser.add_argument("--dataset", default="toy", type=str) # toy, charis_abp, charis_ecg, charis_icp
    parser.add_argument("--test_intra", default=False, action="store_true")
    parser.add_argument("--saved_model", default=None, type=str)
    parser.add_argument("--output_dir", default="experiment_outputs", type=str)
    parser.add_argument("--n_iter", default=28251, type=int) # 100, 000 and 50,000 iterations for UCI dataset
    parser.add_argument("--verbosity", default=250, type=int)
    parser.add_argument("--batch_size", default=1000, type=int) # equals to total number of data points
    args = parser.parse_args()
    run(args)