"""using sentence BERT encoder output as features, simple linear regression to fit gold label
1.save and load embeddings for dataset train, dev and test 
2.hyper-parameters set as args, easy to adjust during training
3.input: dataset name, output correlation, pass hyper-parameters
edit date: 2021-04-28
"""
import os
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch import nn
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
assert pyro.__version__.startswith('1.6.0')
pyro.set_rng_seed(1)

import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from util_data import embed2torch, get_labels, eval_correlation
from sbert_embedding import save_sbert_features
from util_quantile import summary, check_Z, eval_uncertainty

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 0.2).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 0.1).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

def run_blr(args, pd_preds=False):
    # load training data
    data_list = save_sbert_features(dataset_name=args.dataset, returnFL=True)
    train_e1, train_e2, train_labels = data_list[0]
    # generate features as torch.tensor
    x_data, y_data = embed2torch(train_e1, train_e2, train_labels)
    try: 
        dev_e1, dev_e2, dev_labels = data_list[1]
        x_data_dev, y_data_dev = embed2torch(dev_e1, dev_e2, dev_labels)
    except IndexError:
        x_data_dev = None
        print("There isn't dev set.")
    try: 
        test_e1, test_e2, test_labels = data_list[2]  
        x_data_test, y_data_test = embed2torch(test_e1, test_e2, test_labels)
    except IndexError:
        x_data_test = None
        print("There isn't test set.")

    # model
    model = BayesianRegression(x_data.size()[-1], 1)
    guide = AutoDiagonalNormal(model)

    adam = pyro.optim.Adam({"lr": args.lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    num_iterations = args.num_iterations

    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))

    guide.requires_grad_(False)
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))
    # guide.quantiles([0.25, 0.5, 0.75])

    # evaluation
    predictive = Predictive(model, guide=guide, num_samples=800,
                            return_sites=("linear.weight", "obs", "_RETURN"))

    # Evaluation on dev set
    if not x_data_dev is None:
        print(x_data_dev.size(), y_data_dev.size())
        samples = predictive(x_data_dev)
        pred_summary = summary(samples)
        mu = pred_summary["_RETURN"]
        y = pred_summary["obs"]
        
        y_pred = mu["mean"]
        eval_pearson, eval_spearman = eval_correlation(dev_labels, y_pred)
        y_pred = y["mean"]
        eval_pearson, eval_spearman = eval_correlation(dev_labels, y_pred)

        # uncertainty eval
        uncertainty_metrics_mu = eval_uncertainty(dev_labels, mu)
        uncertainty_metrics_y = eval_uncertainty(dev_labels, y)
    if not x_data_test is None:
        print(x_data_test.size(), y_data_test.size())
        samples = predictive(x_data_test)
        pred_summary = summary(samples)
        mu = pred_summary["_RETURN"]
        y = pred_summary["obs"]

        y_pred = mu["mean"]
        eval_pearson, eval_spearman = eval_correlation(test_labels, y_pred)
        y_pred = y["mean"]
        eval_pearson, eval_spearman = eval_correlation(test_labels, y_pred)

        # uncertainty eval
        uncertainty_metrics_mu = eval_uncertainty(test_labels, mu)
        uncertainty_metrics_y = eval_uncertainty(test_labels, y) 
    
    if pd_preds:
        predictions = pd.DataFrame({
            "mu_mean": mu["mean"],
            "mu_perc_5": mu["5%"],
            "mu_perc_95": mu["95%"],
            "y_mean": y["mean"],
            "y_perc_5": y["5%"],
            "y_perc_95": y["95%"],
            "true_gdp": y_data_dev,
        })
        return predictions

def run_gpr(args):
    # load training data
    data_list = save_sbert_features(dataset_name=args.dataset, returnFL=True)
    train_e1, train_e2, train_labels = data_list[0]
    # generate features as torch.tensor
    x_data, y_data = embed2torch(train_e1, train_e2, train_labels)
    try: 
        dev_e1, dev_e2, dev_labels = data_list[1]
        x_data_dev, y_data_dev = embed2torch(dev_e1, dev_e2, dev_labels)
    except IndexError:
        x_data_dev = None
        print("There isn't dev set.")
    try: 
        test_e1, test_e2, test_labels = data_list[2]  
        x_data_test, y_data_test = embed2torch(test_e1, test_e2, test_labels)
    except IndexError:
        x_data_test = None
        print("There isn't test set.")

    X = x_data[:5000]
    y = y_data[:5000]
    print(X.size(), y.size())
    # initialize the inducing inputs
    N = 100  # mini-batch size
    Xu = torch.ones(N, X.size()[-1]) * 0.1
    print(Xu.size())
    # initialize the kernel and model
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim=X.size()[-1], variance=torch.tensor(5.0),
                            lengthscale=torch.tensor(10.0))
    # we increase the jitter for better numerical stability
    sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, noise=torch.tensor(1.), jitter=1.0e-5)

    # the way we setup inference is similar to above
    optimizer = torch.optim.Adam(sgpr.parameters(), lr=args.lr)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = args.num_iterations
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(sgpr.model, sgpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (i + 1, loss.item()))
    
    print("inducing points:\n{}".format(sgpr.Xu.data.numpy()))
    
    # Evaluation
    model = sgpr
    with torch.no_grad():
        mean, cov = model(x_data_dev, full_cov=True, noiseless=False)
    sd = cov.diag().sqrt()  # standard deviation at each input point x
    print(mean.size(), sd.size()) # what we want is the mean and sd
    # Evaluation on dev set
    if not x_data_dev is None:
        print(x_data_dev.size(), y_data_dev.size())
        with torch.no_grad():
            mean, cov = model(x_data_dev, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        # print(mean.size(), sd.size()) # what we want is the mean and sd
        eval_pearson, eval_spearman = eval_correlation(dev_labels, mean)

        # uncertainty evaluation
        preds = check_Z(mean, sd)
        uncertainty_metrics_y = eval_uncertainty(dev_labels, preds)

    if not x_data_test is None:
        print(x_data_test.size(), y_data_test.size())
        with torch.no_grad():
            mean, cov = model(x_data_test, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        eval_pearson, eval_spearman = eval_correlation(test_labels, mean)

        # uncertainty evaluation
        preds = check_Z(mean, sd)
        uncertainty_metrics_y = eval_uncertainty(test_labels, preds)

def run_lr(args):
    # load training data
    data_list = save_sbert_features(dataset_name=args.dataset, returnFL=True)
    train_e1, train_e2, train_labels = data_list[0]
    # generate features as torch.tensor
    x_data, y_data = embed2torch(train_e1, train_e2, train_labels)
    try: 
        dev_e1, dev_e2, dev_labels = data_list[1]
        x_data_dev, y_data_dev = embed2torch(dev_e1, dev_e2, dev_labels)
    except IndexError:
        x_data_dev = None
        print("There isn't dev set.")
    try: 
        test_e1, test_e2, test_labels = data_list[2]  
        x_data_test, y_data_test = embed2torch(test_e1, test_e2, test_labels)
    except IndexError:
        x_data_test = None
        print("There isn't test set.")

    # Regression model
    linear_reg_model = PyroModule[nn.Linear](x_data.size()[-1], 1)

    # Define loss and optimize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=args.lr)
    num_iterations = args.num_iterations

    for j in range(num_iterations):
        # initialize gradients to zero
        optim.zero_grad()
        # run the model forward on the data
        y_pred = linear_reg_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()

        if (j + 1) % 150 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))


    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())

    # Evaluation on dev set
    if not x_data_dev is None:
        print(x_data_dev.size(), y_data_dev.size())
        y_pred = linear_reg_model(x_data_dev).squeeze(-1).detach().numpy()
        # print(y_pred)
        eval_pearson, eval_spearman = eval_correlation(dev_labels, y_pred)
    if not x_data_test is None:
        print(x_data_test.size(), y_data_test.size())
        y_pred = linear_reg_model(x_data_test).squeeze(-1).detach().numpy()
        eval_pearson, eval_spearman = eval_correlation(test_labels, y_pred) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Pyro Linear Regression Model Training")
    parser.add_argument('--net_type', default='lr', type=str, help='model')
    parser.add_argument('--dataset', default='stsb', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    parser.add_argument('--num_iterations', default=1500, type=int, help='training steps')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()

    if args.net_type == "lr":
        run_lr(args)
    elif args.net_type == "blr":
        run_blr(args)
    elif args.net_type == "gpr":
        run_gpr(args)
    
