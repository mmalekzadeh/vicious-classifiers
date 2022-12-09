import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn.functional as F
from math import exp
from torch import nn
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import torchvision 
# from . import ssim 
from piq import ssim, SSIMLoss
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc 
import torchvision.transforms as transforms

def generate_img_adv(args, model, data_loader, adversary, quick=True):
    for batch_id, (_images, _labels) in enumerate(data_loader):                                
        _images, _labels = _images.to(args.device), _labels.to(args.device)                                     
        _images_adv = adversary.run_standard_evaluation(_images, _labels, bs=args.adv_bs, return_labels=False)
        if batch_id == 0:                    
            preds_cln = model(_images).max(1)[1]
            preds_adv = model(_images_adv).max(1)[1]
            labels = _labels
            images_adv = _images_adv
        else:                    
            preds_cln = torch.cat((preds_cln, model(_images).max(1)[1]), dim=0) 
            preds_adv = torch.cat((preds_adv, model(_images_adv).max(1)[1]), dim=0)            
            labels = torch.cat((labels, _labels), dim=0)  
            images_adv = torch.cat((images_adv, _images_adv), dim=0)  
        if quick and (batch_id+1)*args.adv_bs >= args.n_adv_per_epoch:
            break
    
    correct_pred = labels.eq(preds_cln).to(labels.device)
    fooled_pred = (correct_pred & ~preds_cln.eq(preds_adv)).to(labels.device)                             
    images_adv = images_adv[correct_pred & fooled_pred]
    labels_adv = labels[correct_pred & fooled_pred]            
    return images_adv, labels_adv
##################################################################
def evaluate_acc_target(args, model_F, dataloader, cf_mat=False):
    model_F.eval()
    valid_batch_acc = []
    y_true , y_pred = [], []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):                
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid = model_F(images)                
        predictions = output_valid.max(1)[1]
        current_acc = torch.sum((predictions == labels).float())
        valid_batch_acc.append(current_acc)
        n_samples += len(labels)        
        y_true = y_true + labels.tolist()
        y_pred = y_pred + predictions.tolist()
    acc_clean = (sum(valid_batch_acc)/n_samples)*100
    if cf_mat:      
        cf = confusion_matrix(y_true, y_pred, normalize='true')
        return acc_clean, cf
    return acc_clean


def evaluate_acc_enc_dec_cat(args, model_F, model_G, dataloader, return_outputs = False):
    model_F.eval()
    model_G.eval()       
    valid_batch_mse, valid_batch_acc = [], []
    n_samples = 0
    loss_mse = 0.
    loss_ssim = 0.
    loss_mean = 0.
    if args.dataset == "celeba":  
        all_acc = np.zeros(len(args.attributes))
    else:
        all_acc = np.zeros(1)
    for batch_id, (images, labels) in enumerate(dataloader):                
        with torch.no_grad():
            images, labels = images.to(args.device), labels.to(args.device)
            n_samples += len(labels) 
            logits = model_F(images)  
            
            # if  args.output == "Raw":
            #     outputs = logits
            # elif  args.output == "Tanh":
            #     outputs = torch.tanh(logits)
            # elif args.output == "Sigmoid":
            #     outputs = torch.sigmoid(logits)
            # elif args.output == "Softmax":
            #     outputs = torch.softmax(logits, dim=-1)
            # else:
            #     raise ValueError("Wrong value for args.output") 
            outputs = logits
            
            if args.beta_2 > 0.0: 
                rec_images = model_G(outputs)
            
                loss_mse += nn.MSELoss(reduction="sum")(rec_images, images)
                # loss_mse += nn.L1Loss(reduction="sum")(out_img, images).item()
                # loss_ssim += ssim.MSSSIM(size_average=False, channel=args.n_channels)(out_img, images).sum().item()
                # ssim_loss_fn = ssim
                ssim_loss_fn = SSIMLoss(data_range=1.)
                # loss_ssim += ssim_loss_fn.msssim(out_img, images, size_average=True, normalize="relu").item() 
                # loss_ssim += ssim_loss_fn.ssim(rec_images, images, size_average=True).item() 
                loss_ssim += ssim_loss_fn(images, rec_images)
                if len(labels) < args.n_batch:
                    print("\n****\nWarning\n*****") 
            
            if args.beta_1 > 0.0:
                if args.dataset == "celeba":     
                    for j in range(len(labels[0])):
                        all_acc[j] += torch.sum(((logits[:,j] >= 0.).type_as(labels) == labels[:,j]).float()).detach().cpu().numpy()
                else:            
                    all_acc[0] += torch.sum((logits.max(1)[1] == labels).float()).detach().cpu().numpy()
            
            if return_outputs:
                if batch_id == 0:
                    _rec_images = rec_images.cpu().detach()
                    _rec_labels = labels.cpu().detach()
                else:
                    _rec_images = torch.cat((_rec_images, rec_images.cpu().detach()), dim=0)  
                    _rec_labels = torch.cat((_rec_labels, labels.cpu().detach()), dim=0)  
      
    if args.beta_2 > 0.0:         
        loss_mse = loss_mse/(n_samples*args.image_size*args.image_size*args.n_channels)
        # loss_ssim = 1-(loss_ssim/(n_samples))
        # loss_ssim = 1-(loss_ssim/(n_samples/args.n_batch))
        loss_ssim = (loss_ssim/(n_samples/args.n_batch))
    else:
        # loss_ssim = torch.Tensor([1.]) 
        # loss_mse = torch.Tensor([1.]) 
        loss_ssim = 1. 
        loss_mse = 1.  
    if args.beta_1 > 0.0:
        all_acc = (all_acc/n_samples)
    else:
        all_acc = 1/len(args.attributes) 
    
    if return_outputs:
        return loss_mse, loss_ssim, all_acc, _rec_images, _rec_labels
    
    return loss_mse, loss_ssim, all_acc

def print_dataset_info(args, dataset):    
    print("Data Dimensions: ",dataset[0].shape)
    labels = np.array(dataset[1]).astype(int)
    if len(labels.shape)==1: 
        _unique, _counts = np.unique(labels, return_counts=True)        
        print(np.asarray((_unique, _counts)).T)
    else:
        for i in range(len(labels[0])):
            _unique, _counts = np.unique(labels[:,i], return_counts=True)        
            print(f"Attribute {args.attributes[i]}:\n")    
            print(np.asarray((_unique, _counts)).T)


def imshow(img):
    plt.figure(figsize=(25,8))
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    

def get_aug_set(args):
    AUG_SET =   [transforms.RandomHorizontalFlip(p=1.),
                   transforms.RandomErasing(scale=(.02, 0.2), ratio=(.2, 2.2), value=0, p=1.),
                   transforms.RandomPerspective(distortion_scale=0.5, p=1.),               
                   transforms.RandomAdjustSharpness(sharpness_factor=5, p=1.),
                   transforms.RandomAutocontrast(p=1.),
                   ###
                   #transforms.RandomRotation(degrees=(0, 180)),
                   transforms.Compose([
                       transforms.CenterCrop(size=args.image_size*2//3),
                       transforms.Resize(size=args.image_size)]
                   ),
                #   transforms.Compose([
                #       transforms.Pad(padding=10, fill=.0),
                #       transforms.Resize(size=args.image_size)]
                #   ),                      
                   transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
                #   transforms.RandomAffine(degrees=(1, 2), translate=(0.1, 0.3), scale=(0.75, 1.), shear=0.2)
                   ###
                   #, transforms.Lambda(lambda t: t.add(1).div(2)), # From [-1,1] to [0,1]
                   #transforms.Lambda(lambda t: t.mul(2).sub(1)), # From [0,1] to [-1,1]
                ] 
    return AUG_SET
    
    
    
################ ################ ################
def compute_mah_dist(x, y, covM):
    diff = x - y
    inv_covM = torch.linalg.pinv(covM)
    return torch.sqrt(torch.einsum('nj,jk,nk->n', diff, inv_covM, diff))

def compute_rec_risk(args, model_F, model_G, dataloader, meanD, covM):
    model_F.eval()
    model_G.eval()       
    n_samples = 0
    rec_risk = 0.
    for batch_id, (images, labels) in enumerate(dataloader):                
        with torch.no_grad():
            images, labels = images.to(args.device), labels.to(args.device)
            n_samples += len(labels) 
            logits = model_F(images)  
            rec_images = model_G(logits)
            
            org_x = torch.reshape(images, (images.shape[0],np.prod(images.shape[1:])))
            rec_x = torch.reshape(rec_images, (images.shape[0],np.prod(images.shape[1:])))
            mean_x = torch.reshape(meanD, (1,np.prod(meanD.shape)))
            
            print(org_x.shape,rec_x.shape,mean_x.shape)
            
            md_1 = compute_mah_dist(org_x, rec_x)
            md_2 = compute_mah_dist(org_x, mean_x)
            rec_risk += -torch.log(torch.div(md_1,md_2))
                 
    rec_risk = torch.sum(mah_dist)/n_samples
    return rec_risk