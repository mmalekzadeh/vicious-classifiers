import sys
import copy
import os
import argparse
import numpy as np
import torch
import torchvision 

from torch.utils.data import TensorDataset, DataLoader, TensorDataset
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

############################
from TDRA import datasets, models, utils, trainers
from setting import args_parser


if __name__ == '__main__':
    
    args = args_parser() ## Reading the input arguments (see setting.py)

    if torch.cuda.is_available():
        args.device = "cuda"
        args.gpu = 0
        torch.cuda.set_device(args.gpu)
    
    # ### Optional: For making your results reproducible
    # torch.manual_seed(args.rand_seed)
    # torch.cuda.manual_seed(args.rand_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.rand_seed)


    ## Fetch the datasets
    if args.dataset == "mnist":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
    if args.dataset == "f_mnist":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
    if args.dataset == "cifar10":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
    if args.dataset == "cifar100":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
    if args.dataset == "tinyimgnet":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
    elif args.dataset == "celeba":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)    
        train_labels, valid_labels, test_labels = datasets.prepare_labels(args, train_labels, test_labels, valid_labels)
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    
        pos_weights = torch.Tensor((len(train_labels) - train_labels.sum(0))/train_labels.sum(0))
        # pass
    print("\n*** Dataset's Info") 
    print("Training")
    utils.print_dataset_info(args, (train_images, train_labels))
    # if args.dataset == "celeba" or args.dataset == "mnist":
    print("Validation")
    utils.print_dataset_info(args, (valid_images, valid_labels))
    print("Testing")
    utils.print_dataset_info(args, (test_images, test_labels))       


    ## For logging
    exp_name = args.dataset+"_"+args.output+"/"+\
                str(len(args.attributes))+"_"+str(int(args.beta_1*100))+"_"+str(int(args.beta_2*100))+\
                "_"+str(args.image_size)+"_"+str(int(args.alpha_ssim*100))+"_"+str(int(args.alpha_mse*100))+\
                "_"+str(args.with_aug)+"_"+str(args.model_width)+"_"+str(args.rand_seed)
    ## Models
    if args.dataset == "mnist":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.MNIST_MEAN,
                                    std = datasets.MNIST_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.MNIST_MEAN,
                                    std = datasets.MNIST_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)
    if args.dataset == "f_mnist":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.F_MNIST_MEAN,
                                    std = datasets.F_MNIST_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.F_MNIST_MEAN,
                                    std = datasets.F_MNIST_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)
    elif args.dataset == "cifar10":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.CIFAR10_MEAN,
                                    std = datasets.CIFAR10_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.CIFAR10_MEAN,
                                    std = datasets.CIFAR10_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)    
    elif args.dataset == "cifar100":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.CIFAR100_MEAN,
                                    std = datasets.CIFAR100_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.CIFAR100_MEAN,
                                    std = datasets.CIFAR100_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)    

    elif args.dataset == "tinyimgnet":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.TINYIMGNET_MEAN,
                                    std = datasets.TINYIMGNET_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.TINYIMGNET_MEAN,
                                    std = datasets.TINYIMGNET_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)

    elif args.dataset == "celeba":
        model_F = models.WideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,
                                    mean =  datasets.CELEBA_MEAN,
                                    std = datasets.CELEBA_STD,
                                    num_input_channels=args.n_channels)    
        model_G = models.DecWideResNet(num_classes = len(args.attributes),
                                    width = args.model_width,                                
                                    mean =  datasets.CELEBA_MEAN,
                                    std = datasets.CELEBA_STD,
                                    num_input_channels=args.n_channels,
                                    input_type = args.output)
        

    model_F.to(args.device)
    model_G.to(args.device)
    summary(model_F, input_size=(args.n_channels, args.image_size, args.image_size), device=args.device)                
    summary(model_G, input_size=(len(args.attributes),), device=args.device)                

    save_dir = args.root_dir+"/results/"+exp_name+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    if args.dataset == "celeba":
        model_F, model_G = trainers.train_enc_dec_cat(args, model_F, model_G, train_dataset, pos_weights, save_path=save_dir)  
    else:
        model_F, model_G = trainers.train_enc_dec_cat(args, model_F, model_G, train_dataset, pos_weights=None, save_path=save_dir)      
    
    model_F.load_state_dict(torch.load(save_dir+"best_model_F.pt", map_location=torch.device(args.device)))         
    model_G.load_state_dict(torch.load(save_dir+"best_model_G.pt", map_location=torch.device(args.device))) 
    test_data_loader = DataLoader(TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                            batch_size=args.n_batch, shuffle=False, drop_last=False)  
    mse_err, ssim_err, att_accs = utils.evaluate_acc_enc_dec_cat(args, model_F, model_G, test_data_loader)                                
    print("Test MSE: {:.4f}".format(mse_err))
    d_range = test_images.max()-test_images.min()
    PSNR = 10.0 * np.log10(d_range**2/mse_err)
    print("Test PSNR: {:.2f}".format(PSNR)) 
    print("Test SSIM: {:.4f}".format(1-ssim_err)) 
    print("Test Accs:", att_accs*100) 
    print("Mean Test Accs:", np.mean(att_accs)*100) 

    def compute_meanD_and_covM(train_images, valid_images):
        data = np.append(train_images, valid_images, axis=0)
        meanD = data.mean(0).copy()
        data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        print(data.shape)
        covM = np.cov(data, rowvar=False)
        return meanD, covM
    meanD, covM = compute_meanD_and_covM(train_images, valid_images)
    meanD = torch.Tensor(meanD).to(args.device)
    covM = torch.Tensor(covM).to(args.device)

    def compute_mah_dist(x, y, inv_covM):
        diff = x - y
        # torch.sqrt(torch.dot(diff[0], torch.matmul(inv_covM, diff[0]))))
        return torch.sqrt(torch.einsum('nj,jk,nk->n', diff, inv_covM, diff))

    def compute_rec_risk(args, model_F, model_G, dataloader, meanD, covM):
        model_F.eval()
        model_G.eval()       
        n_samples = 0
        rec_risk = 0.

        inv_covM = torch.linalg.pinv(covM)

        for batch_id, (images, labels) in enumerate(dataloader):                
            with torch.no_grad():
                images, labels = images.to(args.device), labels.to(args.device)
                n_samples += len(labels) 
                logits = model_F(images)  
                rec_images = model_G(logits)
                
                org_x = torch.reshape(images, (images.shape[0],np.prod(images.shape[1:])))
                rec_x = torch.reshape(rec_images, (rec_images.shape[0],np.prod(rec_images.shape[1:])))
                mean_x = torch.reshape(meanD, (np.prod(meanD.shape),))

                # utils.imshow(torchvision.utils.make_grid(images[:10], nrow=10))        
                # utils.imshow(torchvision.utils.make_grid(rec_images[:10], nrow=10))        
                # utils.imshow(torchvision.utils.make_grid(meanD, nrow=10))                    
                md_xr = compute_mah_dist(org_x, rec_x, inv_covM)
                md_xm = compute_mah_dist(org_x, mean_x, inv_covM)            
                rec_risk += torch.div(md_xm, md_xr)
                
        rec_risk = torch.sum(rec_risk)/n_samples
        return rec_risk    

    test_data_loader = DataLoader(TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                            batch_size=args.n_batch, shuffle=False, drop_last=True)  
    rec_risk = compute_rec_risk(args, model_F, model_G, test_data_loader, meanD, covM)        
    print("# Epsilon: {:.4f}".format(rec_risk.item()))
    
