import os
import json
from time import perf_counter

import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn

import baselines
import parser
from utils import BATCH_SIZE, print
import utils

dtype = utils.set_dtype(torch.cuda.is_available())
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")
print("pytorch device", device)

PRETRAIN_WEIGHTS_DIR = "pretrained_weights"

def pretrain(net, A_val, y_batch_val, args):
    # rkan3 TODO: load pretrain weights
    # o.w. pretrain and save weights

    # if args.DATASET == 'xray':
        # raise NotImplementedError("rkan3 pretrain xray")
    # elif args.DATASET == 'mnist':
    #     pass
    # elif args.DATASET == 'retino':
        # raise NotImplementedError("rkan3 pretrain retino")

    dpath = check_pretrain_weights_dir(args)
    loaded_net, loaded_z, success = load_weights(dpath, args)
    if success:
        return loaded_net, loaded_z
    else:
        print("  Pretraining")
        net, z_stats = _pretrain(net, A_val, y_batch_val, args)
        save_weights(net, z_stats, dpath, args)
        return net, z_stats


def check_pretrain_weights_dir(args):
    if args.DEMO == 'True':
        num_shots = utils.parse_pretrain_args(args.configs_fname)[0]["num_shots"]
        dpath = os.path.join(PRETRAIN_WEIGHTS_DIR, f"{args.DATASET}_demo", f"shots{num_shots}")
    else:
        dpath = os.path.join(PRETRAIN_WEIGHTS_DIR, args.DATASET)

    if not os.path.isdir(dpath):
        os.makedirs(dpath)

    return dpath

def get_pretrain_filepaths(dpath, args):
    loss_type = utils.parse_pretrain_args(args.configs_fname)[0]["loss"]
    n_meas = args.NUM_MEASUREMENTS
    net_path = os.path.join(dpath, f"net_{loss_type}_{n_meas}.pth")
    z_path = os.path.join(dpath, f"latent_codes_{loss_type}_{n_meas}.pth")
    return net_path, z_path

def load_weights(dpath, args):
    
    t0 = perf_counter()

    net_path, z_path = get_pretrain_filepaths(dpath, args)

    # check paths
    if not (os.path.isfile(net_path) and os.path.isfile(z_path)):
        return None, None, False

    # load pretrained nn
    net = utils.init_dcgan(args)
    if CUDA:
        net.load_state_dict(torch.load(net_path))
    else:
        net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
    net.train()

    # load optimized initial latent codes
    if CUDA:
        z = torch.load(z_path)
    else:
        z = torch.load(z_path, map_location=torch.device('cpu'))

    print(f"Loaded net and z from {net_path} \n\tand {z_path}\n\tin {round(perf_counter() - t0, 2)} seconds")

    return net, z, True

def save_weights(net, z, dpath, args):

    t0 = perf_counter()

    net_path, z_path = get_pretrain_filepaths(dpath, args)

    torch.save(net.state_dict(), net_path)
    torch.save(z, z_path)

    print(f"Saved net and z into {net_path} \n\tand {z_path}\n\tin {round(perf_counter() - t0, 2)} seconds")

def get_data(args, num_shots):
    compose = utils.define_compose(args.NUM_CHANNELS, args.IMG_SIZE)

    if args.DEMO == 'True':
        image_direc = f'pretrain_data/{args.DATASET}_demo/shots{num_shots}/'
    else:
        image_direc = f'pretrain_data/{args.DATASET}/shots{num_shots}/'

    dataset = utils.ImageFolderWithPaths(image_direc, transform = compose)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=num_shots)

    data, _, im_path = next(iter(dataloader))
    # print(f"loaded data {data.shape} from {im_path}")

    return data.to(device)


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

def get_pretrain_loss(loss_type):
    if loss_type == "MMD-multiscale":
        return lambda x, y: MMD(x, y, kernel="multiscale")
    elif loss_type == "MMD-rbf":
        return lambda x, y: MMD(x, y, kernel="rbf")
    elif loss_type == "L2":
        return nn.MSELoss(reduction="mean")
    else:
        raise NotImplementedError(f"pretrain loss {loss_type}. expected MMD or L2")


def _pretrain(net, A_val, y_batch_val, args):
    """
    For training our low shot models, we used the Adam optimizer [18] with
    a learning rate of 10−3
    for 50, 000 iterations.


    Our low shot models followed a similar procedure (Adam optimizer)
    where we first optimized the latent
    space for 1250 iterations with a learning rate of 5 ∗ 10−2

    Then the parameters and latent space were
    jointly optimized for 350 iterations with a learning rate of 10−4.
    """

    t0_full = perf_counter()

    ## TODO: pretrain estimates for theta and Zs
    pretrain_args, latent_codes_args = utils.parse_pretrain_args(args.configs_fname)

    num_shots = pretrain_args["num_shots"]
    zs = torch.zeros(num_shots*args.Z_DIM).type(dtype).view(num_shots, args.Z_DIM,1,1)
    zs.data.normal_().type(dtype) #init random input seed
    zs.requires_grad_(requires_grad=True)
    zs = zs.to(device)
    # print(f"Created Zs {zs.shape} with requires_grad = {zs.requires_grad}")

    loss_type = pretrain_args["loss"] 
    loss_fn = get_pretrain_loss(loss_type = loss_type)
    loss_list = []
    data = get_data(args, num_shots=pretrain_args["num_shots"])
    optimizer = torch.optim.Adam(
    [
        {'params': net.parameters()},
        {'params': zs},
    ],
    lr=pretrain_args["learning_rate"]
    )

    print(f"\tPretraining with {num_shots} shots with {loss_type} loss")
    niters = pretrain_args["num_iterations"]
    t0 = perf_counter()
    for i in range(niters):
        # each iteration computes loss
        # across entire few-shot dataset
        optimizer.zero_grad()
        Gz = net(zs)
        loss = loss_fn(Gz, data)
        loss.backward() # backprop
        optimizer.step()

        if i % 5000 == 0:
            print(f"  Iteration {i} of {niters}")
            print(f"    loss: {loss.item()}")
            print(f"    time per 5k iter: {round(perf_counter() - t0, 2)}s")
            t0 = perf_counter()


    delta_t = round(perf_counter() - t0_full, 2)
    print(f"\t_pretrain total time {delta_t}s")

    return net, zs

def estimate_initial_latent_codes(net, zs,  A_val, y_batch_val, args):

    t0_full = perf_counter()

    pretrain_args, latent_codes_args = utils.parse_pretrain_args(args.configs_fname)
    num_shots = pretrain_args["num_shots"]

    ## TODO: get a sample from multivariate gaussian distribution fits for Z's
    z_numpy = zs.detach().cpu().numpy().reshape((num_shots, args.Z_DIM))
    # print(z_numpy.shape)
    mean = np.mean(z_numpy, axis=0)
    cov = np.cov(z_numpy, rowvar=0)

    # return net, (mean, cov)

    z_init = multivariate_normal.rvs(mean=mean, cov=cov, random_state=0).reshape((1, args.Z_DIM, 1, 1))
    # print(f"Finished sampling from multivariate gaussian fit of Zs, {z_init.shape}")


    # ## TODO: find optimal latent codes
    y = torch.FloatTensor(y_batch_val).type(dtype).to(device) # init measurements y
    A = torch.FloatTensor(args.A).type(dtype).to(device)       # init measurement matrix A

    z_optimal = torch.from_numpy(z_init).type(dtype)
    z_optimal.requires_grad_(True)
    z_optimal = z_optimal.to(device)
    # print(f"Created Z {z_optimal.shape} with requires_grad = {z_optimal.requires_grad}")


    print(f"\tInitializing Z with {zs.shape[0]} shots with MSE loss")
    L2NormSquared = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam([z_optimal], lr=latent_codes_args["learning_rate"])
    niters = latent_codes_args["num_iterations"]
    t0 = perf_counter()
    for i in range(niters):
        optimizer.zero_grad()

        # calculate data fidelity loss
        # = 0.5 * L2NormSquared(y - A*G(z))
        # = 0.5 * MSELoss
        G = net(z_optimal)
        AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
        loss = torch.mul(L2NormSquared(AG, y), 0.5)
        
        # backprop
        loss.backward()
        optimizer.step()

        # if i % 50 == 0:
        #     print(f"  Iteration {i} of {niters}")
        #     print(f"    loss: {loss.item()}")
        #     print(f"    time/iter: {round(perf_counter() - t0, 2)}s")
        #     t0 = perf_counter()

    delta_t = round(perf_counter() - t0_full, 2)
    print(f"\testimate_initial_latent_codes total time {delta_t}s")
    # ## Note: we skip joint optimization
    # # instead use the LR optimization scheme

    return net, z_optimal.detach()

def main():

    t0_full = perf_counter()


    ## copied from comp_sensing.py
    CONFIG_FILENAME = "configs.json"
    CONFIG_FILENAME = "configs_test.json"
    args = parser.parse_args(CONFIG_FILENAME)
    print(args)

    NUM_MEASUREMENTS_LIST, ALG_LIST, VARIANCE_LIST = utils.convert_to_list(args)
    dataloader = utils.get_data(args) # get dataset of images

    for num_meas in NUM_MEASUREMENTS_LIST:
        args.NUM_MEASUREMENTS = num_meas 
        print("NUM_MEASUREMENTS", args.NUM_MEASUREMENTS)
        

        # init measurement matrix
        A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
        args.A = A

        # A_list = []
        # for i in range(10):
        #     # init measurement matrix
        #     A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
        #     args.A = A
        #     A_list.append(A)
        # for i in range(len(A_list)-1):
        #     A1 = A_list[i]
        #     A2 = A_list[i+1]
        #     print("  ", np.array_equal(A1, A2))

        for alg in ALG_LIST:
            args.ALG = alg
            if args.ALG == "csdip" and args.PRETRAIN:
                pretrain_args, _ = utils.parse_pretrain_args(args)
                args.ALG += "+" + pretrain_args["num_shots"]
        
            for _, (batch, _, im_path) in enumerate(dataloader):

                for var in VARIANCE_LIST:
                    # eta_sig = 0 # set value to induce noise 
                    eta_sig = np.sqrt(var)
                    args.STDEV = eta_sig
                    args.VARIANCE = var
                    eta = utils.get_noise(eta_sig, args)
                    print(f"stdev = {eta_sig}; var = {var}")

                    x = batch.view(1,-1).cpu().numpy() # define image
                    y = np.dot(x,A) + eta

                    if utils.recons_exists(args, im_path): # to avoid redundant reconstructions
                        continue
                    NEW_RECONS = True

                    if alg != 'csdip':
                        raise ValueError(f"pretraining not defined for algorithm {alg}. expected csdip")
                    

                    net = utils.init_dcgan(args)
                    net.to(device)
                    net, z = pretrain(net, A, y, args)
    print("Total Time: ", round(perf_counter() - t0_full, 2), "seconds")


if __name__ == "__main__":
    main()




