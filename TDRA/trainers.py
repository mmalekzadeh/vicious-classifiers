import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from . import utils 
# from . import ssim 
from piq import ssim, SSIMLoss
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def imshow(img):
    plt.figure(figsize=(10,8))
    # img = img / 2 + 0.5     # from [-1,1] to [0,1]
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_data_loader(args, dataset, train=True, batch_size=None):
        # if args.dataset == "utk_face":
        #     if train:
        #         _x = dataset[0][args.valid_share:]
        #         _y = dataset[1][args.valid_share:]
        #         _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
        #         _xy = TensorDataset(_x, _y)            
        #         data_loader = DataLoader(_xy, batch_size=args.n_batch,
        #                             shuffle=True, drop_last=True)            
        #     else:
        #         _x = dataset[0][:args.valid_share]
        #         _y = dataset[1][:args.valid_share]
        #         _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
        #         _xy = TensorDataset(_x, _y)            
        #         data_loader = DataLoader(_xy, batch_size=args.n_batch,
        #                             shuffle=True, drop_last=False)
        #     return data_loader
        # elif args.dataset == "celeba" or args.dataset == "mnist" or args.dataset == "cifar10" or args.dataset == "cifar100":
        if batch_size == None:
            batch_size = args.n_batch
        if train:                
            _xy = TensorDataset(torch.Tensor(dataset[0][0]), torch.Tensor(dataset[0][1]).long())
            data_loader = DataLoader(_xy, batch_size=batch_size, shuffle=True, drop_last=True)            
        else:
            _xy = TensorDataset(torch.Tensor(dataset[1][0]), torch.Tensor(dataset[1][1]).long())     
            data_loader = DataLoader(_xy, batch_size=batch_size, shuffle=True, drop_last=True) 
        return data_loader

def train_model_standard(args, model_F, dataset, save_path=None):        
        optimizer_F = optim.Adam(model_F.parameters(), lr=args.lr)                    
        ce_loss_fn = nn.CrossEntropyLoss()
        train_epoch_loss, train_epoch_acc, valid_epoch_acc = [], [], []
        best_valid_acc = 0.
        for epoch in range(args.n_epochs):            
            ################### Training ###################
            if args.with_aug:
                trainloader = get_data_loader(args, dataset, train=True, batch_size=args.n_batch//2)
            else:
                trainloader = get_data_loader(args, dataset, train=True)
            train_batch_loss, train_batch_acc = [], []            
            for batch_id, (images, labels) in enumerate(trainloader):                
                images, labels = images.to(args.device), labels.to(args.device)
                ###################
                if args.with_aug:
                    AUG_SET = utils.get_aug_set(args)
                    # images = torch.stack([
                    #             transforms.Compose([
                    #                 AUG_SET[-2],
                    #                 AUG_SET[np.random.randint(len(AUG_SET)-2)],
                    #                 AUG_SET[-1]])(img) 
                    #             for img in images])
                    aug_images = torch.stack([AUG_SET[np.random.randint(len(AUG_SET))](img) 
                                        for img in images])
                    images = torch.cat((images, aug_images), dim=0)
                    labels = torch.cat((labels, labels), dim=0)
                ###################
                model_F.train()
                optimizer_F.zero_grad()                 
                outputs = model_F(images)
                loss = ce_loss_fn(outputs, labels)            
                loss.backward()
                optimizer_F.step()
                ###################
                train_batch_loss.append(loss.item())
                train_batch_acc.append(torch.mean((outputs.max(1)[1] == labels).float()))
            train_epoch_loss.append(sum(train_batch_loss)/len(train_batch_loss))
            train_epoch_acc.append(sum(train_batch_acc)/len(train_batch_acc)*100)
            ################### Validation ################### 
            validloader =  get_data_loader(args, dataset, train=False)
            acc = utils.evaluate_acc_target(args, model_F, validloader)
            valid_epoch_acc.append(acc)
            print("_________ Epoch "+str(epoch+1)+" _________")
            if save_path:
                if  valid_epoch_acc[-1] > best_valid_acc:  
                    best_valid_acc = valid_epoch_acc[-1] 
                    torch.save(model_F.state_dict(), save_path+"best_model_F.pt")
                    print("**** Best Valid Acc {:.2f}".format(best_valid_acc))
            print("- Train Loss: {:.5f}, \n- Train Acc: {:.2f}".format(train_epoch_loss[-1],train_epoch_acc[-1]))           
            print("- Valid Acc: {:.2f}".format(valid_epoch_acc[-1])) 
        return model_F
 
def train_enc_dec_cat(args,  model_F, model_G, dataset, pos_weights=None,  save_path=None):        
     
        optimizer_F = optim.Adam(model_F.parameters(), lr=args.lr)  
        optimizer_G = optim.Adam(model_G.parameters(), lr=args.lr)                    
         
        mse_loss_fn = nn.HuberLoss(delta=args.hl_delta)  
        # mse_loss_fn = nn.MSELoss()   
        # mse_loss_fn =  nn.L1Loss()
        
        # ssim_loss_fn = ssim
        ssim_loss_fn = SSIMLoss(data_range=1.)
        ce_loss_fn = nn.CrossEntropyLoss()
        
        train_epoch_loss_ce, train_epoch_loss_mse, train_epoch_loss_ssim = [], [], []
        
        train_epoch_loss, train_epoch_acc, valid_epoch_accs,  = [], [], []
        valid_epoch_mse, valid_epoch_ssim, valid_epoch_mean = [], [], []
        
        best_valid_loss = 1e5     
        
        for epoch in range(args.n_epochs):            
            ## Training
            if args.with_aug:
                trainloader = get_data_loader(args, dataset, train=True, batch_size=args.n_batch//2)
            else:
                trainloader = get_data_loader(args, dataset, train=True)
            
            train_batch_loss_ce, train_batch_loss_mse, train_batch_loss_ssim = [], [], []
            
            train_batch_loss, train_batch_acc = [], []                        
            train_disc_batch_loss = []
            
            for batch_id, (images, labels) in enumerate(trainloader):                
                
                images, labels = images.to(args.device), labels.to(args.device)
                ###################
                if args.with_aug:
                    AUG_SET = utils.get_aug_set(args)
                    aug_images = torch.stack([AUG_SET[np.random.randint(len(AUG_SET))](img) 
                                        for img in images])
                    aug_images = torch.clamp(aug_images, min=0.0, max=1.0)
                    images = torch.cat((images, aug_images), dim=0)
                    labels = torch.cat((labels, labels), dim=0)
                ###################
                
                #### Training model
                model_F.train()
                model_G.train()
                optimizer_F.zero_grad()   
                optimizer_G.zero_grad()   
                
                logits = model_F(images) 
                  
                loss = 0.
                if args.beta_1 > 0.0:
                    if args.dataset == "celeba":
                        for j in range(len(labels[0])):
                            bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights[j])
                            _loss_ = bce_loss_fn(logits[:,j], labels[:,j].type_as(logits))
                            loss += args.beta_1*_loss_
                        loss = loss/len(labels[0])
                        train_batch_loss_ce.append(loss.item())
                    else:
                        _loss_ = ce_loss_fn(logits, labels)
                        train_batch_loss_ce.append(_loss_.item())
                        loss += args.beta_1*_loss_
                
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
                    ## MSE
                    _loss_ = mse_loss_fn(rec_images, images)
                    train_batch_loss_mse.append(_loss_.item())
                    loss += args.beta_2*args.alpha_mse*_loss_
                    ## SSIM
                    # _loss_ = 1-ssim_loss_fn.ssim(rec_images, images)
                    _loss_ = ssim_loss_fn(images, rec_images)
                    train_batch_loss_ssim.append(_loss_.item())
                    loss += args.beta_2*args.alpha_ssim*_loss_
                    
                    # loss += -args.beta_2*args.alpha_ssim*ssim_loss_fn.msssim(out_img, images, normalize="relu")
                    
                ## Total Variation Loss
                # w_variance = nn.L1Loss()(out_img[:,:,:,:-1] , out_img[:,:,:,1:])  
                # h_variance = nn.L1Loss()(out_img[:,:,:-1,:] , out_img[:,:,1:,:])
                # loss += (w_variance + h_variance) 
                
                loss.backward()                
         
                optimizer_F.step()
                optimizer_G.step()  
                ##################
                
                train_batch_loss.append(loss.item())
                
                avg_acc_t = 0.
                if args.dataset == "celeba":     
                    for j in range(len(labels[0])):
                        avg_acc_t += torch.mean(((logits[:,j] >= 0.).type_as(labels) == labels[:,j]).float())
                        train_batch_acc.append(avg_acc_t/len(labels[0]))
                else:
                    avg_acc_t = torch.mean((logits.max(1)[1] == labels).float())
                    train_batch_acc.append(avg_acc_t)
                
                
            train_epoch_loss.append(sum(train_batch_loss)/len(train_batch_loss))
            if args.beta_1 > 0.0:
                train_epoch_loss_ce.append(sum(train_batch_loss_ce)/len(train_batch_loss_ce))
            if args.beta_2 > 0.0:
                train_epoch_loss_mse.append(sum(train_batch_loss_mse)/len(train_batch_loss_mse))
                train_epoch_loss_ssim.append(sum(train_batch_loss_ssim)/len(train_batch_loss_ssim))
                
            train_epoch_acc.append(sum(train_batch_acc)/len(train_batch_acc)*100)
            
            ## Validation 
            validloader =  get_data_loader(args, dataset, train=False)
            mse_err, ssim_err, att_accs = utils.evaluate_acc_enc_dec_cat(args, model_F, model_G, validloader)            
            valid_epoch_mse.append(mse_err)
            valid_epoch_ssim.append(ssim_err)
            valid_epoch_accs.append(att_accs)
            
            print("_________ Epoch: ", epoch+1)
            if save_path:
                if args.beta_1 > 0.0 and args.beta_2 > 0.0:  
                    current_result =  (1-np.mean(valid_epoch_accs[-1])) + args.alpha_ssim*valid_epoch_ssim[-1] + args.alpha_mse*valid_epoch_mse[-1]
                elif args.beta_1 > 0.0:
                    current_result =  (1-np.mean(valid_epoch_accs[-1])) 
                else:
                    current_result = args.alpha_ssim*valid_epoch_ssim[-1]  + args.alpha_mse*valid_epoch_mse[-1]  
                if current_result < best_valid_loss:
                    best_valid_loss = current_result 
                    torch.save(model_F.state_dict(), save_path+"best_model_F.pt")
                    torch.save(model_G.state_dict(), save_path+"best_model_G.pt")
                    print("**** Best Acc on Epoch {} is {:.4f}".format(epoch+1, best_valid_loss))                        
            print("Train Loss : {:.5f}, \nTrain Acc: {:.2f}".format(train_epoch_loss[-1],
                                                            train_epoch_acc[-1]))    
            if args.beta_1 > 0.0:
                print("--Loss CE : {:.5f}".format(train_epoch_loss_ce[-1]))                               
            if args.beta_2 > 0.0:
                print("--Loss MSE {:.8f}, \n--Loss SSIM: {:.8f}".format(train_epoch_loss_mse[-1],
                                                            train_epoch_loss_ssim[-1]))                               
            print("Valid MSE: {:.8f}".format(valid_epoch_mse[-1])) 
            print("Valid SSIM: {:.8f}".format(1-valid_epoch_ssim[-1])) 
            print("Valid Accs:",valid_epoch_accs[-1]*100) 
            print("Valid Accs Mean:",np.mean(valid_epoch_accs[-1]*100)) 
            
            if args.beta_2 > 0.0:
                model_F.eval()
                model_G.eval()  
                for batch_id, (images, labels) in enumerate(validloader):    
                    images, labels = images.to(args.device), labels.to(args.device)
                    logits = model_F(images) 
                    utils.imshow(torchvision.utils.make_grid(images[:10], nrow=10))
                    
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
                        
                    rec_images = model_G(outputs)
                    utils.imshow(torchvision.utils.make_grid(rec_images[:10], nrow=10))
                    break
        return model_F, model_G 