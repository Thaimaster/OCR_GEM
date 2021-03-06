# Using multiprocessing.Processing
# main & train func run in 1 process with major process

from multiprocessing import Process
from multiprocessing import freeze_support
from multiprocessing.process import AuthenticationString # for sercurity
from time import sleep, asctime

import matplotlib.pyplot as plt
#import psutil

import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

from torch.autograd import Variable
from torch.utils import data
import os
from tqdm import tqdm
import multiprocessing
import collections

from dataset import IC15Loader, lock, mp_diff_imgs_paths
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util
import config
from focalloss import *
#============== Function ===================#
def check_stop(list_proc, list_stop_f):
    #terminate process when stop flag=True
    print('checking stop:')
    for i, e in enumerate(list_stop_f):  
        #print(i,e)
        if list_stop_f[i].is_set()==True:
            print('end process id: ' ,list_proc[i].pid)  
            list_proc[i].terminate() 
            list_proc.pop(i)
            list_stop_f.pop(i)
            
lock = multiprocessing.Lock()

def ohem_single(score, n_pos, neg_mask):
    if n_pos == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = neg_mask
        return selected_mask
    
    neg_num = neg_mask.view(-1).sum()
    neg_num = (min(n_pos * config.max_neg_pos_ratio, neg_num)).to(torch.int)
    
    if neg_num == 0:
        selected_mask = neg_mask
        return selected_mask

    neg_score=torch.masked_select(score,neg_mask)*-1
    value,_=neg_score.topk(neg_num)
    threshold=value[-1]

    selected_mask= neg_mask*(score<=-threshold)
    return selected_mask

def set_threshold(images_loss):
    value = images_loss.values()
    value = sorted(value, reverse=False)
    # print("------------------",len(value), type(value))
    # print(int(len(value)*0.6))
    threshold = value[int(len(value)*0.6)]
    # print("threshold",threshold)
    return threshold


def ohem_image(batch_idx, imgs, epoch, images_loss):
    status = False
    
    if epoch == 0:
        status = True
        return status, imgs
    
    threshold = set_threshold(images_loss)
    if images_loss[batch_idx]>=threshold:
        status = True
        return status, imgs

    else:
        return status, imgs 
     

def ohem_batch(neg_conf, pos_mask, neg_mask):
    selected_masks = []
    for img_neg_conf,img_pos_mask,img_neg_mask in zip(neg_conf,pos_mask,neg_mask):
        n_pos=img_pos_mask.view(-1).sum()
        selected_masks.append(ohem_single(img_neg_conf, n_pos, img_neg_mask))

    selected_masks = torch.stack(selected_masks, 0).to(torch.float)

    return selected_masks

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)
    
    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= config.pixel_conf_threshold] = 0
    pred_text[pred_text >  config.pixel_conf_threshold] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
#======================Class multi Processing==============================#
# 
class Multi_train(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        # must call this before anything else
        super().__init__(*args, **kwargs)

        # then any other initialization
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        self.hyperparams = self.Hyperparams()
        #self.params_train = None
    def run(self):
        self.main(self.hyperparams)
        
    def __getstate__(self):
        """called when pickling - this hack allows subprocesses to 
           be spawned without the AuthenticationString raising an error"""
        state = self.__dict__.copy()
        conf = state['_config']
        if 'authkey' in conf: 
            #del conf['authkey']
            conf['authkey'] = bytes(conf['authkey'])
        return state

    def __setstate__(self, state):
        """for unpickling"""
        state['_config']['authkey'] = AuthenticationString(state['_config']['authkey'])
        self.__dict__.update(state)
        
    #============= Main and Train Func ===============# 
    def train(self, train_loader,data_loader, images_loss, model, criterion, optimizer, epoch, start_epoch, writer=None, val_loader=None):
        import config 
        cls_loss_lambda=config.pixel_cls_loss_weight_lambda
        link_loss_lambda=config.pixel_link_loss_weight

        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        loss_cls = AverageMeter()
        running_metric_text = runningScore(2)
        
        # info training
        # _losses loss=losses.avg,
        # loss_cls=loss_cls.avg,
        # acc=score_text['Mean Acc'],
        # iou_t=score_text['Mean IoU'])

        # device=torch.device('cuda:0')
        device=torch.device("cuda:0") # with graph Model 2 GPU
        end = time.time()
        
        for batch_idx, (imgs, img_paths, cls_label, cls_weight, link_label, link_weight) in tqdm(enumerate(train_loader)):
            data_time.update(time.time() - end)
            
            # print(img_paths)
            # if img_paths[0] in train_loader.dataset.difficult_imgs_paths:
            #   print("ohem")
              # data_loader.remove_difficult_images(img_paths[0])
            imgs=imgs.to(device)
            # imgs=imgs.to('cuda:1')# Graph 2 GPU
            cls_label, cls_weight, link_label, link_weight = cls_label.to(
                device), cls_weight.to(device), link_label.to(device), link_weight.to(device)
            
            link_label = link_label.transpose(2,3).transpose(1,2) # [b, 8, h, w]
            link_weight = link_weight.transpose(2,3).transpose(1,2) # [b, 8, h, w]

            # outputs=model(imgs)

            # pixel_cls_logits = outputs[:, 0:2, :, :]
            # pixel_link_logits = outputs[:, 2:, :, :]

            pixel_cls_logits,pixel_link_logits=model(imgs)                       

            pos_mask=(cls_label>0)
            neg_mask=(cls_label==0)

            train_mask=pos_mask+neg_mask
            pos_logits=pixel_cls_logits[:,1,:,:]

            pixel_cls_loss=criterion(pos_logits,pos_mask.to(torch.float),train_mask.to(torch.float))
            # for text class loss
            # pixel_cls_loss=F.cross_entropy(pixel_cls_logits,pos_mask.to(torch.long),reduce=False)
            

            # criterion = FocalLoss(gamma=2, alpha=0.25)
            # criterion_link = FocalLoss2(gamma=2, alpha=0.25)

            # pixel_cls_loss = criterion(pixel_cls_logits, pos_mask.to(torch.long))
            
            pixel_cls_scores=F.softmax(pixel_cls_logits,dim=1)
            pixel_neg_scores=pixel_cls_scores[:,0,:,:]
            # print("p", pixel_neg_scores)
            # exit()

            selected_neg_pixel_mask=ohem_batch(pixel_neg_scores,pos_mask,neg_mask)
            
            n_pos=pos_mask.view(-1).sum()
            n_neg=selected_neg_pixel_mask.view(-1).sum()

            pixel_cls_weights=(cls_weight+selected_neg_pixel_mask).to(torch.float)

            cls_loss=(pixel_cls_loss*pixel_cls_weights).view(-1).sum()/(n_pos+n_neg)
            
            # for link loss
            if n_pos==0:
                link_loss=(pixel_link_logits*0).view(-1).sum()
                shape=pixel_link_logits.shape
                pixel_link_logits_flat=pixel_link_logits.contiguous().view(shape[0],2,8,shape[2],shape[3])
            else:
                shape=pixel_link_logits.shape
                pixel_link_logits_flat=pixel_link_logits.contiguous().view(shape[0],2,8,shape[2],shape[3])
                link_label_flat=link_label

                pixel_link_loss=F.cross_entropy(pixel_link_logits_flat,link_label_flat.to(torch.long),reduce=False)
                # pixel_link_loss = criterion_link(pixel_link_logits_flat, link_label_flat.to(torch.long))

                def get_loss(label):
                    link_mask=(link_label_flat==label)
                    link_weight_mask=link_weight*link_mask.to(torch.float)
                    n_links=link_weight_mask.reshape(-1).sum()
                    loss=(pixel_link_loss*link_weight_mask).reshape(-1).sum()/n_links
                    return loss
                
                neg_loss = get_loss(0)
                pos_loss = get_loss(1)


                neg_lambda=1.0
                link_loss=pos_loss+neg_loss*neg_lambda
            
            
            # loss_item=cls_loss_lambda*list_loss+link_loss_lambda*link_loss
            
            loss = cls_loss_lambda*cls_loss+link_loss_lambda*link_loss
        
            images_loss.update({batch_idx:loss.item()}) 
            OHEM_img_threshold = config.OHEM_img_threshold
            # print(loss.item() ,"LLLLL",OHEM_img_threshold)
            if epoch != start_epoch:
                OHEM_img_threshold = config._set_OHEM_img_threshold(images_loss)
            # calculer acc , mean iou , ...
            score_text = cal_text_score(F.softmax(pixel_cls_logits,dim=1)[:,1,:,:], pos_mask, cls_label>-1, running_metric_text) 
            # print(img_paths[0])
            # print("Overall Acc", score_text['Overall Acc'])
            # print("Mean Acc", score_text['Mean Acc'])
            # print("FreqW Acc", score_text['FreqW Acc'])

            if  loss.item() > OHEM_img_threshold :    
                # print(img_paths[0], "lossL:", cls_loss.item(), "threshold",OHEM_img_threshold)
                updated_dict = {img_path: 1 for img_path in img_paths}
                # print("update",updated_dict)
                with lock: 
                  mp_diff_imgs_paths.update(updated_dict)
                  # print(" after added:", mp_diff_imgs_paths)
              

            loss_cls.update(cls_loss.cpu().item(), imgs.size(0))
            
            losses.update(loss.cpu().item(), imgs.size(0))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # del loss
            # # and free cache
            # torch.cuda.empty_cache()

            # score_text = cal_text_score(F.softmax(pixel_cls_logits,dim=1)[:,1,:,:], pos_mask, cls_label>-1, running_metric_text)
            
            
            batch_time.update(time.time() - end)
            end = time.time()
            # print("---------------------------batch_idx --------------------- : ", batch_idx)
            
            if (val_loader != None):
                if ((batch_idx % 20 == 0) and (batch_idx != 0)): #or (batch_idx == (len(train_loader)-1)
                    # if batch_idx%40==0:
                    #     grid=torchvision.utils.make_grid(imgs[:2,:,:,:],4,normalize=True)
                    #     writer.add_image("image",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                    #     pos_score=pixel_cls_scores[:,1:,:,:]
                    #     grid=torchvision.utils.make_grid(pos_score[:2,:,:,:],4)
                    #     writer.add_image("pos_score",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')
                    #     grid=torchvision.utils.make_grid(pos_mask[:2,None,:,:].to(torch.float),4,normalize=True)
                    #     writer.add_image("pos_mask",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                    #     grid=torchvision.utils.make_grid(link_label[:2,0:1,:,:].to(torch.float),4,normalize=True)
                    #     writer.add_image("link_label_0",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                    #     link_score=F.softmax(pixel_link_logits_flat,dim=1)[:2,1,0:1,:,:]*pos_mask[:2,None,:,:].to(torch.float)
                    #     grid=torchvision.utils.make_grid(link_score,4,normalize=True)
                    #     writer.add_image("link_score_0",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                    # writer.add_scalar("cls_loss",cls_loss.cpu().item(),len(train_loader)*epoch+batch_idx)
                    # writer.add_scalar("link_loss",link_loss.cpu().item(),len(train_loader)*epoch+batch_idx)

                    output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f}| Loss_cls: {loss_cls:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(train_loader),
                        bt=batch_time.avg,
                        total=batch_time.avg * batch_idx / 60.0,
                        eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                        loss=losses.avg,
                        loss_cls=loss_cls.avg,
                        acc=score_text['Mean Acc'],
                        iou_t=score_text['Mean IoU'])
                    print(output_log)            
                
                
                sys.stdout.flush()
        if (val_loader != None):
            end_epoch = 'losses:{losses:.3f} loss_cls:{loss_cls:.3f} acc:{acc:.3f} IoU_acc:{IoU_acc:.3f}'.format(
                    losses = losses.avg, 
                    loss_cls = loss_cls.avg, 
                    acc = score_text['Mean Acc'], 
                    IoU_acc = score_text['Mean IoU'])
        else:
            end_epoch = 'val_loss:{losses:.3f} val_loss_cls:{loss_cls:.3f} val_acc:{acc:.3f} val_IoU_acc:{IoU_acc:.3f}'.format(
                    losses = losses.avg, 
                    loss_cls = loss_cls.avg, 
                    acc = score_text['Mean Acc'], 
                    IoU_acc = score_text['Mean IoU'])
        print(end_epoch)
        
        return (losses.avg, score_text['Mean Acc'], score_text['Mean IoU'])
        
      
    def main(self, args):
        print('start training in process id:%s ' %(os.getpid()))
        if args.checkpoint == '':
            args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
        if args.pretrain:
            if 'synth' in args.pretrain:
                args.checkpoint += "_pretrain_synth"
            else:
                args.checkpoint += "model_japan_1280_v2"

        print(('checkpoint path: %s'%args.checkpoint))
        print(('init lr: %.8f'%args.lr))
        print(('schedule: ', args.schedule))
        print(args)
        sys.stdout.flush()

        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)

        writer=SummaryWriter(args.checkpoint)

        kernel_num=18
        start_epoch = 0
        #####
        #
        #
        #--------Split Training and Validate dataset
        #####
        #load all data
        data_loader = IC15Loader(is_transform=True, img_size=args.img_size)
        train_loader1 = torch.utils.data.DataLoader(
            data_loader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True)
        ####
        #dataset = CustomDatasetFromCSV(my_path)
        bz = 1
        validation_split = .2
        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(data_loader)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        # split here
        train_loader = torch.utils.data.DataLoader(data_loader, batch_size=bz, num_workers=0,
                                                  sampler=train_sampler, drop_last=False,
                                                        pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(data_loader, batch_size=bz, num_workers=0,
                                                        sampler=valid_sampler, drop_last=False,
                                                        pin_memory=True)
        print('train: ',len(train_loader))
        print('train ban ?????u: ',len(train_loader1))
        print('validation: ',len(validation_loader))
        # Usage Example:
        #for batch_index, (faces, labels) in enumerate(train_loader):

        #print('stop')
        #asd[0] =1
        ####
        if args.arch == "resnet50":
            model = models.resnet50(pretrained=True, num_classes=kernel_num)
        elif args.arch == "resnet101":
            model = models.resnet101(pretrained=True, num_classes=kernel_num)
        elif args.arch == "resnet152":
            model = models.resnet152(pretrained=True, num_classes=kernel_num)
        elif args.arch == "vgg16":
            model = models.vgg16(pretrained=False,num_classes=kernel_num)
            # model = models.ModelParallelVGG16(pretrained=False,num_classes=kernel_num) # graph 2 GPU
            
        
        model = torch.nn.DataParallel(model).cuda() # graph 2 GPU comment
        model.train()

        # if hasattr(model.module, 'optimizer'):
        #     optimizer = model.module.optimizer
        
        #use Graph 2GPU
        if hasattr(model, 'optimizer'):
            optimizer = model.optimizer
        else:
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.90, weight_decay=5e-4)
            # NOTE ???????????????momentum????????????????????????????????????0.99?????????crossentropy????????????.
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.92, weight_decay=5e-4)

        title = 'icdar2015'
        if args.pretrain:
            print('Using pretrained model.')
            assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.pretrain)
            print("loaded ",list(checkpoint['state_dict'].keys())[:10])
            # load model Colab to graph 2 GPU
            # print("backbone",model.state_dict().keys())
            # checkpoint_dict = {}
            # for k,v in checkpoint['state_dict'].items(): 
                
            #     k = k.replace('module.', "")
            #     checkpoint_dict[k.replace('backbone', "module1")] = v
            #     checkpoint_dict[k.replace('backbone', "module2")] = v
            # print("edit",list(checkpoint_dict.keys())[:10])
            print("model", list(model.state_dict().keys())[:10])
            # exit()
            
            # # model.load_state_dict({k.replace('module.','module1') :v for k,v in checkpoint.items()})
            # model.load_state_dict(checkpoint_dict,strict=False)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
            # save_checkpoint({
            #             'epoch': start_epoch,
            #             'state_dict': model.state_dict(),
            #             'lr': args.lr,
            #             'optimizer' : optimizer.state_dict(),
            #         }, checkpoint=args.checkpoint,filename='test_%d.pth'%start_epoch)
            # exit()
        elif args.resume:
            print('Resuming from checkpoint.')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print('Training from scratch.')
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
        images_loss = {}
        # data_plot = images_loss.values()
        # import matplotlib.pyplot as plt
        # plt.plot(data_plot)
        # plt.ylabel('Loss plot')
        # plt.show()
        
        for epoch in range(start_epoch, args.n_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            print(('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr'])))
            
            train_loss, train_te_acc, train_te_iou = self.train(train_loader,data_loader,images_loss, model, dice_loss, optimizer, epoch,start_epoch,writer)
            val_loss, val_te_acc, val_te_iou = self.train(validation_loader,data_loader,images_loss, model, dice_loss, optimizer, epoch,start_epoch,writer, val_loader=True)            
            
            #update infor training       
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_acc.append(train_te_acc)
            self.val_acc.append(val_te_acc)
            
            if (epoch %5 ==0 and epoch != 0) : #
                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'lr': args.lr,
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint=args.checkpoint,filename='checkpoint_%d.pth'%epoch)
            if (epoch == (args.n_epoch-1)):
                self.show_loss()
                self.show_acc()

            # logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou])
        
        logger.close()
        writer.flush()
        writer.close()

        
    def stop_training(self, stop_flag):
        #event user want stop training
        # need set flag of list_stop_f
        stop_flag.set()
        return
        
    #Visualize
    # plot losses & accuracy      
    def show_loss(self):
        plt.figure(figsize=(5, 2.7))
        plt.plot(self.train_losses, label=str(self.train_losses))  # Plot some data on the (implicit) axes.
        plt.plot(self.val_losses, label=str(self.val_losses))  # etc.
        plt.xlabel('iteration')
        plt.ylabel('value')
        plt.title(" Loss Diagram")
        plt.legend()
        #plt.show()
        
        #save fig
        os.makedirs('save_fig/loss', exist_ok=True)
        plt.savefig(f'save_fig/loss/fig_{os.getpid()}_{time.time()}.png')
        
        #save csv
        np.savetxt(f'save_fig/loss/train_losses_{os.getpid()}.csv', 
                   self.train_losses,
                   delimiter =", ", 
                   fmt ='% s')
        np.savetxt(f'save_fig/loss/val_losses_{os.getpid()}.csv', 
                   self.val_losses,
                   delimiter =", ", 
                   fmt ='% s')
    def show_acc(self):
        plt.figure(figsize=(5, 2.7))
        plt.plot(self.train_acc, label=str(self.train_acc))  # Plot some data on the (implicit) axes.
        plt.plot(self.val_acc, label=str(self.val_acc))  # etc.
        plt.xlabel('iteration')
        plt.ylabel('value')
        plt.title(" Accuracy Diagram")
        plt.legend()
        #plt.show()
        
        #save fig
        os.makedirs('save_fig/acc', exist_ok=True)
        plt.savefig(f'save_fig/acc/fig_{os.getpid()}_{time.time()}.png')
        
        #save csv
        np.savetxt(f'save_fig/loss/train_acc{os.getpid()}.csv', 
                   self.train_losses,
                   delimiter =", ", 
                   fmt ='% s')
        np.savetxt(f'save_fig/loss/val_acc{os.getpid()}.csv', 
                   self.val_losses,
                   delimiter =", ", 
                   fmt ='% s')

      
    class Hyperparams(object):
        def __init__(self, arch='vgg16', img_size=2480, 
                        n_epoch=737, schedule=[150], 
                        batch_size=16, lr=1e-4, 
                        resume=None, pretrain='/content/drive/MyDrive/ocr/Pretrain_model/model_SGD_e735_5e-4.pth', 
                        checkpoint='/content/drive/MyDrive/ocr/checkpoint/', *args, **kwargs):
            #super().__init__(*args, **kwargs)
            
            self.arch       = arch
            self.img_size   = img_size
            self.n_epoch    = n_epoch
            self.schedule   = schedule
            self.batch_size = batch_size
            self.lr         = lr
            self.resume     = resume
            self.pretrain   = pretrain
            self.checkpoint = checkpoint
            
            
            
        # def __call__(self, arch='resnet50', img_size=2480, schedule=[150], batch_size=16, 
                            # lr=1e-4, resume=None, pretrain=None, checkpoint='checkpoint'):
            # self.arch       = arch
            # self.img_size   = img_size
            # self.schedule   = schedule
            # self.batch_size = batch_size
            # self.lr         = lr
            # self.resume     = resume
            # self.pretrain   = pretrain
            # self.checkpoint = checkpoint
        def __iter__(self):
            print('iter')
            return iter((self.arch, self.img_size, self.n_epoch, self.schedule, self.batch_size, 
                            self.lr, self.resume, self.pretrain, self.checkpoint))   
            
        def __str__(self):
            template = '{0.arch} {0.img_size} {0.n_epoch} {0.schedule} {0.batch_size} {0.lr} {0.resume} {0.pretrain} {0.checkpoint}'
            return template.format(self)
    
##=========================Testing===================##
def test():
    
    #print('------params: ',mt.params)
    #pid_child = multiprocessing.Manager().list()
    #pid_parent = multiprocessing.Manager().list()
    print('start test-----')
    list_proc = []
    list_stop_f = []

    for i in range(1):
        stop_flag = multiprocessing.Event()            
        list_stop_f.append(stop_flag)
       
        # creat new process
        p1 = Multi_train(daemon=False)
        
        #creat params object
        params = p1.hyperparams
            
        params.arch = 'vgg16'
        params.checkpoint = '/content/drive/MyDrive/ocr/checkpoint/'
        params.batch_size = 20
        params.pretrain = '/content/drive/MyDrive/ocr/Pretrain_model/model_SGD_e735_5e-4.pth'
        params.n_epoch = 737
        # print(p1.hyperparams)
        #print('p1.hyperparams.arch: ',p1.hyperparams.arch)
        #print('p1.Hyperparams.arch: ',p1.Hyperparams.arch)
        #print('params from user:%d \n ------------%s----------' %(i, params))
        #sleep(100)

        #asd[0] =1
        print()

        list_proc.append(p1)
        # p1.start()
        # p1.join()
        print(p1)
        
        # Detach train process here so main can return.
        sleep(1)
    for p in list_proc:
        p.start()
    for p in list_proc:
        p.join()
    print ('major main end reached on', asctime())
    print('\n')
        
        # # stop event
        # if i==1: 
            # p1.stop_training(stop_flag)
    # # for debug
    # print(' check event: ')
    # for i, e in enumerate(list_stop_f):
        # print(i, e.is_set())

    # while True:
    # #check every 5s   
    # # cant check is_alive child process of child process
        # check_stop(list_proc, list_stop_f)
        
        # for p in list_proc:
            # print('check PID list_proc:', p,  p.is_alive())
   
        # sleep(10)
        # # print('check pid after process')
        # # tasklist=['python.exe']
        # # out=[]
        # # for proc in psutil.process_iter():
            # # if any(task in proc.name() for task in tasklist):
                # # print([{'pid' : proc.pid, 'name' : proc.name()}])

if __name__ == '__main__': 

    freeze_support()
    test()

#Worked!!!!!!!!!!!!!
