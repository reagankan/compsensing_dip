import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
import time
import sys
from utils import print
from pretrain import pretrain, estimate_initial_latent_codes

args = parser.parse_args('configs.json') 
# args = parser.parse_args('configs_test.json') 
# args = parser.parse_args('configs_15shots.json')

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)
se = torch.nn.MSELoss(reduction='none').type(dtype)

BATCH_SIZE = 1
EXIT_WINDOW = 51
loss_re, recons_re = utils.init_output_arrays(args)

def dip_estimator(args):
    def estimator(A_val, y_batch_val, args):

        total_t0 = time.perf_counter()
        print(f"Starting DIP estimation at time {round(total_t0, 2)}")
        if args.NO_LR:
            print(">> skipping LR")

        y = torch.FloatTensor(y_batch_val).type(dtype) # init measurements y
        A = torch.FloatTensor(A_val).type(dtype)       # init measurement matrix A

        mu, sig_inv, tvc, lrc = utils.get_constants(args, dtype)

        for j in range(args.NUM_RESTARTS):

            t0 = time.perf_counter()
            print(f" Starting restart {j+1} of {args.NUM_RESTARTS} at time {round(t0, 2)}")

            net = utils.init_dcgan(args)

            # rkan3 TODO: load pretrain weights
            # o.w. pretrain and save weights
            if args.PRETRAIN:
                net, zs = pretrain(net, A_val, y_batch_val, args)
                net, z = estimate_initial_latent_codes(net, zs, A_val, y_batch_val, args)

            else:
                z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
                z.data.normal_().type(dtype) #init random input seed
                
            if CUDA:
                net.cuda() # cast network to GPU if available
            
            # rkan3: RMSProp, lr, momentum (used by both papers, src=DIP)
            # solves inverse problem using un/pre-trained network
            optim = torch.optim.RMSprop(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0)
            loss_iter = []
            recons_iter = [] 

            print(f"Reconstructing with {args.NUM_ITER} iters")
            for i in range(args.NUM_ITER):

                optim.zero_grad()

                # calculate measurement loss || y - A*G(z) ||
                G = net(z)
                AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
                y_loss = torch.mean(torch.sum(se(AG,y),dim=1))

                # calculate total variation loss 
                tv_loss = (torch.sum(torch.abs(G[:,:,:,:-1] - G[:,:,:,1:]))\
                            + torch.sum(torch.abs(G[:,:,:-1,:] - G[:,:,1:,:]))) 
            
                # if needed, calculate learned regularization loss
                if args.NO_LR:
                    total_loss = y_loss + tvc*tv_loss # total loss for iteration i
                else:
                    layers = net.parameters()
                    layer_means = torch.cat([layer.mean().view(1) for layer in layers])
                    lr_loss = torch.matmul(layer_means-mu,torch.matmul(sig_inv,layer_means-mu))
                    total_loss = y_loss + lrc*lr_loss + tvc*tv_loss # total loss for iteration i
                 
                # stopping condition to account for optimizer convergence
                if i >= args.NUM_ITER - EXIT_WINDOW: 
                    recons_iter.append(G.data.cpu().numpy())
                    loss_iter.append(total_loss.data.cpu().numpy())
                    if i == args.NUM_ITER - 1:
                        idx_iter = np.argmin(loss_iter)

                total_loss.backward() # backprop
                optim.step()

            # save reconstruction with smallest loss
            # across all optimization iterations
            recons_re[j] = recons_iter[idx_iter]       
            loss_re[j] = y_loss.data.cpu().numpy()

            print(f"  Finished restart {j+1} of {args.NUM_RESTARTS}")
            print(f"  Best Loss {loss_re[j]}")
            print(f"  Time {round(time.perf_counter() - t0, 2)} seconds")
            print()

        # return the reconstruction with smallest loss
        # across all restarts/optimization iterations
        idx_re = np.argmin(loss_re,axis=0)
        x_hat = recons_re[idx_re]
        print(f"Finished DIP in {round(time.perf_counter() - total_t0)} seconds")

        return x_hat

    return estimator
