import numpy as np
import pickle as pkl
import os
import parser
import numpy as np

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines as baselines 
import time

def main():

    t0_full = time.perf_counter()

    NEW_RECONS = False

    # args = parser.parse_args('configs_test.json')
    args = parser.parse_args('configs_15shots.json')
    # args = parser.parse_args('configs_50shots.json')
    # args = parser.parse_args('configs_10shots.json')
    # args = parser.parse_args('configs.json')
    # args = parser.parse_args('configs_100shots.json')
    print(args)

    NUM_MEASUREMENTS_LIST, ALG_LIST, VARIANCE_LIST = utils.convert_to_list(args)

    dataloader = utils.get_data(args) # get dataset of images

    for num_meas in NUM_MEASUREMENTS_LIST:
        args.NUM_MEASUREMENTS = num_meas 
        print("NUM_MEASUREMENTS", args.NUM_MEASUREMENTS)
        
        # init measurement matrix
        A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
        args.A = A

        for alg in ALG_LIST:
            args.ALG = alg
            if args.ALG == "csdip" and args.PRETRAIN:
                pretrain_args, _ = utils.parse_pretrain_args(args.configs_fname)
                args.ALG += "+" + str(pretrain_args["num_shots"])
        
            for var in VARIANCE_LIST:

                for _, (batch, _, im_path) in enumerate(dataloader):

                    print("im_path", im_path)

                    # eta_sig = 0 # set value to induce noise 
                    eta_sig = np.sqrt(var)
                    args.STDEV = eta_sig
                    args.VARIANCE = var
                    eta = utils.get_noise(eta_sig, args)
                    print(f"stdev = {eta_sig}; var = {eta_sig**2}")

                    x = batch.view(1,-1).cpu().numpy() # define image
                    y = np.dot(x,A) + eta

                    if utils.recons_exists(args, im_path): # to avoid redundant reconstructions
                        continue
                    NEW_RECONS = True

                    if alg == 'csdip':
                        estimator = cs_dip.dip_estimator(args)
                    elif alg == 'dct':
                        estimator = baselines.lasso_dct_estimator(args)
                    elif alg == 'wavelet':
                        estimator = baselines.lasso_wavelet_estimator(args)
                    elif alg == 'bm3d' or alg == 'tval3':
                        raise NotImplementedError('BM3D-AMP and TVAL3 are implemented in Matlab. \
                                                    Please see GitHub repository for details.')
                    else:
                        raise NotImplementedError

                    x_hat = estimator(A, y, args)

                    utils.save_reconstruction(x_hat, args, im_path)

    if NEW_RECONS == False:
        print('Duplicate experiment configurations. No new data generated.')
    else:
        print('Reconstructions generated!')

    print("Total Time: ", round(time.perf_counter() - t0_full, 2), "seconds")

if __name__ == "__main__":
    main()
