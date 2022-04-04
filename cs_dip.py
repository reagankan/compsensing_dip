import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
import time
import sys

default_print = print
def print(*args):
    default_print(*args)
    sys.stdout.flush()

args = parser.parse_args('configs.json') 

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

        y = torch.FloatTensor(y_batch_val).type(dtype) # init measurements y
        A = torch.FloatTensor(A_val).type(dtype)       # init measurement matrix A

        mu, sig_inv, tvc, lrc = utils.get_constants(args, dtype)

        for j in range(args.NUM_RESTARTS):
            # rkan3 TODO: load in pretrained weights
            
            t0 = time.perf_counter()
            print(f" Starting restart {j+1} of {args.NUM_RESTARTS} at time {round(t0, 2)}")

            net = utils.init_dcgan(args)

            z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
            z.data.normal_().type(dtype) #init random input seed
            if CUDA:
                net.cuda() # cast network to GPU if available
            
            # rkan3: RMSProp, lr, momentum (used by both papers, src=DIP)
            # solves inverse problem using un/pre-trained network
            optim = torch.optim.RMSprop(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0)
            loss_iter = []
            recons_iter = [] 

            for i in range(args.NUM_ITER):

                optim.zero_grad()

                # calculate measurement loss || y - A*G(z) ||
                G = net(z)
                AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
                y_loss = torch.mean(torch.sum(se(AG,y),dim=1))

                # calculate total variation loss 
                tv_loss = (torch.sum(torch.abs(G[:,:,:,:-1] - G[:,:,:,1:]))\
                            + torch.sum(torch.abs(G[:,:,:-1,:] - G[:,:,1:,:]))) 

                # calculate learned regularization loss
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
