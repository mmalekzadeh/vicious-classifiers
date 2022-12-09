import os
import gc
import numpy as np
import imageio as nd

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision import transforms

 
MNIST_MEAN = (0.13107656) 
MNIST_STD = (0.28947565)
F_MNIST_MEAN = (0.28564665) 
F_MNIST_STD = (0.33854118)
CIFAR10_MEAN = (0.49147478, 0.48220044, 0.4466697)
CIFAR10_STD = (0.24713175, 0.24367353, 0.26168618)
CIFAR100_MEAN = (0.50690794, 0.4865954 , 0.44092864)
CIFAR100_STD = (0.26741382, 0.2564657 , 0.2762759)
CELEBA_MEAN = (0.50634044, 0.42579043, 0.38316137)
CELEBA_STD = (0.2969209, 0.276824 , 0.2767341)
TINYIMGNET_MEAN = (0.48054704, 0.44838083, 0.39780676)  
TINYIMGNET_STD = (0.2540925 , 0.24558398, 0.26044846)

def get_dataset(args):
    """
    Datasets
    """ 
    if args.dataset == "tinyimgnet":
        # !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        # https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet
        save_dir = args.root_dir+"/data/temp/TinyImgNet_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data...")

            path = args.root_dir+'/data/tiny-imagenet-200/'

            def get_id_dictionary():
                id_dict = {}
                for i, line in enumerate(open( path + 'wnids.txt', 'r')):
                    id_dict[line.replace('\n', '')] = i
                return id_dict
            
            def get_class_to_id_dict():
                id_dict = get_id_dictionary()
                all_classes = {}
                result = {}
                for i, line in enumerate(open( path + 'words.txt', 'r')):
                    n_id, word = line.split('\t')[:2]
                    all_classes[n_id] = word
                for key, value in id_dict.items():
                    result[value] = (key, all_classes[key])      
                return result
            
            id_dict = get_id_dictionary()        
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(args.image_size,args.image_size)),
                transforms.ToTensor()                
                ])

            train_data, test_data = [], []
            train_labels, test_labels = [], []    
            for key, value in id_dict.items():
                train_data += [transform(
                    nd.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB')
                    ).numpy()
                     for i in range(500)]
                train_labels_ = np.array([[0]*200]*500)
                train_labels_[:, value] = 1
                train_labels += train_labels_.tolist()

            for line in open( path + 'val/val_annotations.txt'):
                img_name, class_id = line.split('\t')[:2]
                test_data.append(transform(
                    nd.imread( path + 'val/images/{}'.format(img_name) ,pilmode='RGB')
                    ).numpy())
                test_labels_ = np.array([[0]*200])
                test_labels_[0, id_dict[class_id]] = 1
                test_labels += test_labels_.tolist()

            train_data, train_labels = np.array(train_data), np.array(train_labels)
            test_data, test_labels = np.array(test_data), np.array(test_labels)                        
            
            train_labels = train_labels.argmax(1)
            test_labels = test_labels.argmax(1)                    

            ## Saving
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_data, train_labels=train_labels,                                              
                                              test_images=test_data, test_labels=test_labels)
           
            print(len(train_data), len(train_labels))            
            print(len(test_data), len(test_labels))                                              
            
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:90000]
        train_labels = train_valid_labels[:90000]
        valid_images = train_valid_images[90000:] 
        valid_labels = train_valid_labels[90000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)

        return dataset_train, dataset_valid, dataset_test
    

        
    if args.dataset == "celeba":
        ## These are (almost) balanced attributes:
        # {'Attractive': 2, 'Black_Hair': 8, 'Blond_Hair': 9, 'Brown_Hair': 11,
        # 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20,
        # 'Mouth_Slightly_Open': 21, 'Smiling': 31, 'Wavy_Hair': 33, 'Wearing_Lipstick': 36}
        save_dir = args.root_dir+"/data/temp/CelebA_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data...")
            root = args.root_dir+"/data"
            transform=transforms.Compose([
                transforms.Resize(size=(args.image_size,args.image_size)),                
                transforms.ToTensor()])                
            ## For the first time 'download' should be set to True
            dataset_train = CelebA(root=root, split='train', target_type='attr',transform=transform, download=False) 
            dataset_valid = CelebA(root=root, split='valid', target_type='attr',transform=transform) 
            dataset_test  = CelebA(root=root, split='test', target_type='attr',transform=transform) 
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid))    
            valid_images, valid_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
            ## Saving
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              valid_images=valid_images, valid_labels=valid_labels, 
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(valid_images), len(valid_labels))
            print(len(test_images), len(test_labels))                                              
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_images = dataset_npz['train_images']
        train_labels = (dataset_npz['train_labels']).astype(int)
        valid_images = dataset_npz['valid_images']
        valid_labels = (dataset_npz['valid_labels']).astype(int)
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test        
    
    elif args.dataset == "mnist":
        save_dir = args.root_dir+"/data/temp/MNIST_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),                
                transforms.ToTensor()])          
            ## For the first time 'download' should be set to True
            dataset_train = MNIST(root=root, train=True, transform=transform, download=True) 
            dataset_test  = MNIST(root=root, train=False, transform=transform, download=True)
            
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
                        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(test_images), len(test_labels))                                              
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:50000]
        train_labels = train_valid_labels[:50000]
        valid_images = train_valid_images[50000:] 
        valid_labels = train_valid_labels[50000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test
    
    elif args.dataset == "f_mnist":
        save_dir = args.root_dir+"/data/temp/FashionMNIST_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),                
                transforms.ToTensor()])          
            ## For the first time 'download' should be set to True
            dataset_train = FashionMNIST(root=root, train=True, transform=transform, download=True) 
            dataset_test  = FashionMNIST(root=root, train=False, transform=transform, download=True)
            
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
                        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(test_images), len(test_labels))                                              
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:50000]
        train_labels = train_valid_labels[:50000] 
        valid_images = train_valid_images[50000:] 
        valid_labels = train_valid_labels[50000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test
    
    elif args.dataset == "cifar10":
        save_dir = args.root_dir+"/data/temp/CIFAR10_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([transforms.ToTensor()])               
            dataset_train = CIFAR10(root=root, train=True, transform=transform, download=True) 
            dataset_test  = CIFAR10(root=root, train=False, transform=transform, download=True)
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(test_images), len(test_labels))                                              
            print("preprocessing is completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:45000]
        train_labels = train_valid_labels[:45000]
        valid_images = train_valid_images[45000:] 
        valid_labels = train_valid_labels[45000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test 
    
    elif args.dataset == "cifar100":
        save_dir = args.root_dir+"/data/temp/CIFAR100_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([transforms.ToTensor()])                
            ## For the first time 'download' should be set to True
            dataset_train = CIFAR100(root=root, train=True, transform=transform, download=True) 
            dataset_test  = CIFAR100(root=root, train=False, transform=transform, download=True)
            
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
                        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(test_images), len(test_labels))                                              
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:45000]
        train_labels = train_valid_labels[:45000]
        valid_images = train_valid_images[45000:] 
        valid_labels = train_valid_labels[45000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test 
        
##############################################
def prepare_labels(args, train_labels, test_labels, valid_labels=None):
    if args.dataset == "celeba":                 
        train_labels = train_labels[:,args.attributes]
        valid_labels = valid_labels[:,args.attributes]
        test_labels  = test_labels[:,args.attributes]
        return train_labels, valid_labels, test_labels   
