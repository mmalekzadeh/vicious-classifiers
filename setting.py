import argparse

def args_parser():
    r"""
    This file serves as a global enviornment for setting up a simulation.
    """

    parser = argparse.ArgumentParser()
    #### Datasets
    parser.add_argument('--data_preprocessing', default=1, help="For the first time you need to set 1")
    parser.add_argument('--image_size', default=32, help="Size of the Input: e.g., 32x32")
    # ## MNIST
    # parser.add_argument('--dataset', default="mnist", help="The dataset")
    # parser.add_argument('--attributes', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="ATTRIBUTES")
    # parser.add_argument('--n_channels', default=1, help="For RGB is 3, for Grayscale is 1")
    # parser.add_argument('--model_width', default=1, help="the width of WideResNet")

    # ## F_MNIST
    # parser.add_argument('--dataset', default="f_mnist", help="The dataset")
    # parser.add_argument('--attributes', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="ATTRIBUTES")
    # parser.add_argument('--n_channels', default=1, help="For RGB is 3, for Grayscale is 1")
    # parser.add_argument('--model_width', default=5, help="the width of WideResNet")


    # TINYIMGNET
    # parser.add_argument('--dataset', default="tinyimgnet", help="The dataset")
    # parser.add_argument('--attributes', default=list(range(200)), help="ATTRIBUTES")
    # parser.add_argument('--n_channels', default=3, help="For RGB is 3, for Grayscale is 1")
    # parser.add_argument('--model_width', default=5, help="the width of WideResNet")

    ## CelebA
    # attributes_dict = {'Attractive': 2, 'Black_Hair': 8, 'Blond_Hair': 9, 'Brown_Hair': 11,
    #         'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20,
    #         'Mouth_Slightly_Open': 21, 'Smiling': 31, 'Wavy_Hair': 33, 'Wearing_Lipstick': 36}
    # ATTRIBUTES = [31]       
    # ATTRIBUTES = [8, 31] 
    # ATTRIBUTES = [8, 20, 31]                                         
    # ATTRIBUTES = [2, 8, 20, 31]                                               
    # ATTRIBUTES = [2, 8, 19, 20, 31] 
    # ATTRIBUTES = [2, 8, 18, 19, 20, 31]
    # ATTRIBUTES = [2, 8, 18, 20, 31, 36, 39]      
    # ATTRIBUTES = [2, 8, 18, 20, 31, 33, 36, 39]      
    # ATTRIBUTES = [2, 8, 18, 19, 20, 31, 33, 36, 39]      
    # ATTRIBUTES = [2, 8, 11, 18, 19, 20, 31, 33, 36, 39]      
    # ATTRIBUTES = [ 1,  2, 6,  7,  8, 18, 19, 20, 21, 25, 27, 31, 33, 36, 39]
    # ATTRIBUTES = [ 1,  2,  3,  6,  7,  8, 11, 18, 19, 20, 21, 24, 25, 27, 31, 32, 33, 34, 36, 39]
    # ATTRIBUTES = [ 0,1,2,3,5,6,7,8,9,11,12,13,15,18,19,20,21,23,24,25,27,28,31,32,33,34,36,37,38,39]
    # ATTRIBUTES = list(range(40))
    # parser.add_argument('--dataset', default="celeba", help="The dataset")
    # parser.add_argument('--attributes', default=ATTRIBUTES, help="ATTRIBUTES")
    # parser.add_argument('--n_channels', default=3, help="For RGB is 3, for Grayscale is 1")
    # parser.add_argument('--model_width', default=5, help="the width of WideResNet")

    ## CIFAR10
    # parser.add_argument('--dataset', default="cifar10", help="The dataset")
    # parser.add_argument('--attributes', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="ATTRIBUTES")
    # parser.add_argument('--n_channels', default=3, help="For RGB is 3, for Grayscale is 1")
    # parser.add_argument('--model_width', default=5, help="the width of WideResNet")

    ## CIFAR100
    parser.add_argument('--dataset', default="cifar100", help="The dataset")
    parser.add_argument('--attributes', default=list(range(100)), help="ATTRIBUTES")
    parser.add_argument('--n_channels', default=3, help="For RGB is 3, for Grayscale is 1")
    parser.add_argument('--model_width', default=5, help="the width of WideResNet")
    ##### Training
    parser.add_argument('--rand_seed',  default=1, help="Random Seed")
    parser.add_argument('--n_epochs',   default=2, help="Number of Epochs")
    parser.add_argument('--n_batch',    default=250, help="Batch Size for training F")
    parser.add_argument('--output',    default="Raw", help="Raw or Softmax")
    parser.add_argument('--with_aug',   default=True, help="For training with data augmentation")
    ## Optimization
    parser.add_argument('--lr', default=0.001, help="Learning Rate")
    parser.add_argument('--beta_1', default=1., help="Multiplier for the main task")
    parser.add_argument('--beta_2', default=3., help="Multiplier for the secondary task")
    parser.add_argument('--beta_ent', default=0, help="Multiplier for the entropy minimization task")
    parser.add_argument('--alpha_mse', default=5., help="Multiplier for the MSE loss")  
    parser.add_argument('--alpha_ssim', default=1., help="Multiplier for the SSIM loss")

    parser.add_argument('--hl_delta', default=1., help="Huber Loss Delta")
    ## Device
    parser.add_argument('--device', default='cpu', help="When using GPU, set --device='cuda'")
    parser.add_argument('--gpu',    default=None, help="When using GPU, set --gpu=0")
    parser.add_argument('--root_dir',   default="TDRA", help="The root directory for saving data and results.")    
    ###
    args = parser.parse_args()
    return args