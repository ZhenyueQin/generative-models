import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
# from numpy_plotter import plot_a_numpy_array
from general_methods import get_current_time


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

running_time = get_current_time()

for it in range(100000):
    X, _ = mnist.train.next_batch(mb_size, shuffle=True)
    print('X: ', X.shape)
    # plot_a_numpy_array(X[0])
    X = Variable(torch.from_numpy(X))

    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        samples = P(z).data.numpy()[:16]

        if not os.path.exists('original_np_outs/' + running_time + '/'):
            os.makedirs('original_np_outs/' + running_time + '/')

        samples_name = 'original_np_outs/' + running_time + '/' + 'original_samples_' + str(it) + '.out'
        np.savetxt(samples_name, samples, delimiter=',')

if not os.path.exists('original_latent_np_outs/' + running_time + '/'):
    os.makedirs('original_latent_np_outs/' + running_time + '/')

for it in range(100):
    Y, _ = mnist.test.next_batch(64, shuffle=False)
    Y = Variable(torch.from_numpy(Y))
    z_mu, z_var = Q(Y)
    z = sample_z(z_mu, z_var)

    z_np = z.data.numpy()

    # print('z shape: ', z_np.shape)

    latent_z_name = 'original_latent_np_outs/' + running_time + '/' + 'latent_z_' + str(it) + '.out'
    np.savetxt(latent_z_name, z_np, delimiter=',')

