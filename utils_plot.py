# package imports
import numpy as np
import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
# user defined imports
import utils
from skimage.metrics import peak_signal_noise_ratio

class Args(object): # to create an empty argument class
    def __init__(self):
        pass

def construct_arg(args): # define dataset-specific arguments
    if args.DATASET == 'mnist':
        args.IMG_SIZE = 28
        args.NUM_CHANNELS = 1
    elif args.DATASET == 'xray':
        args.IMG_SIZE = 256
        args.NUM_CHANNELS = 1
    elif args.DATASET == 'retino':
        args.IMG_SIZE = 128
        args.NUM_CHANNELS = 3
    else:
        raise NotImplementedErrorFalse
        
    if args.DEMO == 'True':
        args.IMG_PATH = 'data/{0}_demo/'.format(args.DATASET)
        args.MSE_PLOT_PATH = 'plots/{0}_demo_mse.pdf'.format(args.DATASET)
        args.REC_PLOT_PATH = 'plots/{0}_demo_reconstructions'.format(args.DATASET)
    else:
        args.IMG_PATH = 'data/{0}/'.format(args.DATASET)
        args.MSE_PLOT_PATH = 'plots/{0}_mse.pdf'.format(args.DATASET)
        args.REC_PLOT_PATH = 'plots/{0}_reconstructions'.format(args.DATASET)

    return args

def renorm_bm3d(x): # maps [0,256] output from .mat file to [-1,1] for conistency 
    return .0078125*x - 1

def get_plot_data(dataloader, args, old=False): # load reconstructions and compute mse
    RECONSTRUCTIONS = dict()
    MSE = dict()
    PSNR = dict()
    for ALG in args.ALG_LIST:
        args.ALG = ALG
        RECONSTRUCTIONS[ALG] = dict()
        MSE[ALG] = dict()
        PSNR[ALG] = dict()

        for NUM_MEASUREMENTS in args.NUM_MEASUREMENTS_LIST:
            args.NUM_MEASUREMENTS = NUM_MEASUREMENTS
            RECONSTRUCTIONS[ALG][NUM_MEASUREMENTS] = dict()
            MSE[ALG][NUM_MEASUREMENTS] = dict()
            PSNR[ALG][NUM_MEASUREMENTS] = dict()

            ## add 1 more dimension: noise variance
            for var in args.VARIANCE_LIST:
                args.VARIANCE = var
                RECONSTRUCTIONS[ALG][NUM_MEASUREMENTS][var] = list()
                MSE[ALG][NUM_MEASUREMENTS][var] = list()
                PSNR[ALG][NUM_MEASUREMENTS][var] = list()
            
                for _, (batch, _, im_path) in enumerate(dataloader):
                    if args.DATASET == 'retino':
                        batch_ = batch.numpy()[0]
                    else:
                        batch_ = batch.numpy()[0][0]
                    batch_ = batch_.astype(np.float64)
                    rec_path = utils.get_path_out(args,im_path, old=old)
                    if ALG == 'bm3d' or ALG == 'tval3':
                        rec = sio.loadmat(rec_path)['x_hat']
                        rec = renorm_bm3d(rec) # convert [0,255] --> [-1,1] 
                        if args.DATASET == 'retino':
                            rec = np.transpose(rec, (2,0,1))                  
                    else:
                        rec = np.load(rec_path)
                    n = rec.ravel().shape[0]
                    mse = np.power(np.linalg.norm(batch_.ravel() - rec.ravel()),2)/(1.0*n)
                    psnr = peak_signal_noise_ratio(batch_.ravel(), rec.ravel())
                    RECONSTRUCTIONS[ALG][NUM_MEASUREMENTS][var].append(rec)
                    MSE[ALG][NUM_MEASUREMENTS][var].append(mse)
                    PSNR[ALG][NUM_MEASUREMENTS][var].append(psnr)

    METRICS = dict(MSE=MSE, PSNR=PSNR)
    return RECONSTRUCTIONS, METRICS

def set_kwargs(dataset):
    """
    'b' as blue - wavelet

    'g' as green - dct

    'r' as red - csdip+100

    'c' as cyan - csdip+10

    'm' as magenta - csdip+50

    'y' as yellow

    'k' as black - csdip

    'w' as white

    'o' as orange - bm3d

    'v' as lightblue - tval3

    src: https://matplotlib.org/stable/tutorials/colors/colors.html
    """
    if dataset == "mnist":
        KWARGS_DICT = {'csdip':{"fmt":'k-', "label":'csdip', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                  'dct':{"fmt":'g-', "label":'Lasso-DCT', "marker":"s", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                  'wavelet':{"fmt":'b-', "label":'Lasso-DB4', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'bm3d':{"fmt":'o-', "label":'BM3D-AMP', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'tval3':{"fmt":'v-', "label":'TVAL3', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'csdip+10':{"fmt":'c-', "label":'csdip+10', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                   'csdip+50':{"fmt":'m-', "label":'csdip+50', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                   'csdip+100':{"fmt":'r-', "label":'csdip+100', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                  }
    elif dataset == "retino":
        KWARGS_DICT = {'csdip':{"fmt":'k-', "label":'csdip', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                  'dct':{"fmt":'g-', "label":'Lasso-DCT', "marker":"s", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                  'wavelet':{"fmt":'b-', "label":'Lasso-DB4', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'bm3d':{"fmt":'o-', "label":'BM3D-AMP', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'tval3':{"fmt":'v-', "label":'TVAL3', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
                   'csdip+15':{"fmt":'c-', "label":'csdip+15', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
                  }

    for k in KWARGS_DICT:
      del KWARGS_DICT[k]["marker"]
    return KWARGS_DICT

def renorm(x):
    return 0.5*x + 0.5

def plot_format(y_min, y_max, args):
    plt.ylim([y_min,y_max])
    if "ylabel" not in args.__dict__:
        plt.ylabel('Reconstruction Error')
    else:
        plt.ylabel(args.ylabel)
    if "xlabel" not in args.__dict__:
        plt.xlabel('Number of Measurements')
    else:
        plt.xlabel(args.xlabel)
    plt.xticks(args.xticks,args.xticks, rotation=270)
    plt.legend(loc='upper right')

def plot_mse(mse_alg, args, kwargs):
    y_temp = []
    y_error = []
    x_temp = args.NUM_MEASUREMENTS_LIST
    for NUM_MEASUREMENTS in args.NUM_MEASUREMENTS_LIST:
        n = len(mse_alg[NUM_MEASUREMENTS])
        mse = np.mean(mse_alg[NUM_MEASUREMENTS])
        error = np.std(mse_alg[NUM_MEASUREMENTS]) / np.sqrt(1.0*n)
        y_temp.append(mse)
        y_error.append(error)
    # print(y_temp)
    plt.errorbar(x_temp,y_temp,y_error,**kwargs)

def plot_metric(metric_alg, args, kwargs):
    y_temp = []
    y_error = []
    x_temp = metric_alg.keys()
    for metric in metric_alg:
        n = len(metric_alg[metric])
        mean = np.mean(metric_alg[metric])
        error = np.std(metric_alg[metric]) / np.sqrt(1.0*n)
        y_temp.append(mean)
        y_error.append(error)
    # print(y_temp)
    plt.errorbar(x_temp,y_temp,y_error,**kwargs)

figure_height = 5
NUM_PLOT = 5

def set_axes(alg_name, ax):
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_yaxis().set_label_coords(-0.5,0) # (0, 0) is bottom left, (0.5, 0.5) is center, etc.
    ax.set_ylabel(alg_name, fontdict=dict(weight='bold'))

def frame_image(image, cmap = None, axis_dict=None):

    if axis_dict:
        axis = axis_dict.get("axis")
        axis.imshow(image, cmap=cmap)
        axis.set_xticks([], [])
        axis.set_yticks([], [])
        
        label = axis_dict.get("label")
        bold = axis_dict.get("bold")
        if bold:
            axis.set_xlabel(label, fontdict=dict(weight='bold'))
        else:
            axis.set_xlabel(label)
    else:
        frame=plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        # print("frame", type(frame), frame)
        frame = frame.imshow(image, cmap=cmap)

# plot images according to different data format (rgb/grayscale, original/reconstruction, from python/matlab)
def plot_image(image, args, flag, axis_dict=None):
    if flag == 'orig': # for plotting an original image
        if args.NUM_CHANNELS == 3: # for rgb images
            frame_image(renorm(image[0].cpu().numpy().transpose((1,2,0))), axis_dict=axis_dict)
        elif args.NUM_CHANNELS == 1: # for grayscale images
            frame_image(renorm(image[0].cpu().numpy().reshape((args.IMG_SIZE,args.IMG_SIZE))), cmap='gray', axis_dict=axis_dict)
        else:
            raise ValueError('NUM_CHANNELS must be 1 for grayscale or 3 for rgb images.')
    elif flag == 'recons': # for plotting reconstructions
        if args.NUM_CHANNELS == 3: # for rgb images
            if args.ALG[:len("csdip")] == 'csdip':
                frame_image(renorm(image[0][0].transpose((1,2,0))), axis_dict=axis_dict)
            elif args.ALG == 'bm3d':
                frame_image(utils.renorm(np.asarray(image.transpose(1,2,0))), axis_dict=axis_dict)
            elif args.ALG == 'dct' or args.ALG == 'wavelet':
                frame_image(renorm(image.reshape(-1,128,3,order='F').swapaxes(0,1)), axis_dict=axis_dict)
            else:
                raise ValueError('Plotting rgb images is supported only by csdip, bm3d, dct, wavelet.')
        elif args.NUM_CHANNELS == 1: # for grayscale images
            frame_image(renorm(image.reshape(args.IMG_SIZE,args.IMG_SIZE)), cmap='gray', axis_dict=axis_dict)
        else:
            raise ValueError('NUM_CHANNELS must be 1 for grayscale or 3 for rgb images.')
    else:
        raise ValueError('flag input must be orig or recons for plotting original image or reconstruction, respectively.')


### UNUSED FUNCTIONS BELOW ###
# from PIL import Image
def classify(rgb_tuple):
    # will classify pixel into one of following categories based on manhattan distance
    colors = {"red": (255, 0, 0),
              "yellow": (255,255,0),
              "lyellow": (255,255,150),
              "orange": (255,165,0),
              "black": (0,0,0),
              "brown": (132, 85, 27),
              "obrown": (202, 134, 101),
              "bgreen": (12,136,115),
              "green" : (0,255,0),
              "purple": (128,0,128),
              "lpurple": (211,134,248)
              }
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color

white = (255,255,255)
not_converged = ['green','purple','lpurple','bgreen']

def process_rgb(imarray): # update pixel values if not converged
    img = Image.fromarray((imarray*255).astype('uint8'), 'RGB')
    width,height = img.size
    for x in xrange(width):
        for y in xrange(height):
            r,g,b = img.getpixel((x,y))
            col = classify((r,g,b))
            if col in not_converged:
                img.putpixel((x,y), white)
                tt = np.array(img)
    return np.array(img)