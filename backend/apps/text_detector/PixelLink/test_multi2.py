# Using multiprocessing.Processing
# main & train func run in 1 process with major process

from multiprocessing import Process
from multiprocessing import freeze_support
from multiprocessing.process import AuthenticationString # for sercurity
from time import sleep, asctime
import time


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import psutil
import os
import cv2
#from globals import Globals

#================Non daemon====================
# class NoDaemonProcess(multiprocessing.Process):
    # @property
    # def daemon(self):
        # return False

    # @daemon.setter
    # def daemon(self, value):
        # pass


# class NoDaemonContext(type(multiprocessing.get_context())):
    # Process = NoDaemonProcess

# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.pool.Pool):
    # def __init__(self, *args, **kwargs):
        # kwargs['context'] = NoDaemonContext()
        # super(MyPool, self).__init__(*args, **kwargs)
        
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
    def train(self, params_train):
        '''
        Param Train:    train_loader, data_loader, images_loss, model, criterion,
                                optimizer, epoch, start_epoch, writer=None
        '''
        
        print('param  train =', params_train)
        print('training....')
        for _ in range(50):
            self.train_losses.append(np.random.normal((1, )))
        for _ in range(50):
            self.val_losses.append(np.random.normal((1, )))
        #print('train_loss:', self.train_losses)
        #print('pid train:', os.getpid())
        sleep(5)
        
        print ('train end reached on-------------', asctime())
        #  
        print('show train val loss:')
        self.show_loss()
        # can save figure by matplotlib
      
    def main(self, params_main):
        '''
        params_main:    arch, img_size, schedule, batch_size, lr, resume, pretrain, checkpoint
        '''
        
        #global pid
        print('params main', params_main)
        
        self.train(params_main)
        
        print('pid main: ', os.getpid())
        print('end main')
        
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

      
    class Hyperparams():
        def __init__(self, arch='resnet50', img_size=2480, n_epoch=1000, schedule=[150], batch_size=16, 
                            lr=1e-4, resume=None, pretrain=None, checkpoint='checkpoint', *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            self.arch       = arch
            self.img_size   = img_size
            self.n_epoch    = n_epoch
            self.schedule   = schedule
            self.batch_size = batch_size
            self.lr         = lr
            self.resume     = resume
            self.pretrain   = pretrain
            self.checkpoint = checkpoint
            
            #print(Multi_train.hyperparams) 
            
            
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
    #list_mn = multiprocessing.Manager().list()
    list_proc = []
    list_stop_f = []

    for i in range(2):
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
        #print('params.arch:  ', params.arch)
        print(p1.hyperparams.arch)
        #print('p1.hyperparams.arch: ',p1.hyperparams.arch)
        #print('p1.Hyperparams.arch: ',p1.Hyperparams.arch)
        #print('params from user:%d \n ------------%s----------' %(i, params))
        #sleep(100)

        #asd[0] =1
        print()

        list_proc.append(p1)
        #p1.start()       
        print(p1)
        
        # Detach train process here so main can return.
        sleep(1)
    print ('major main end reached on', asctime())
    print('\n')
        
        # # stop event
        # if i==1: 
            # p1.stop_training(stop_flag)
    
    # start all process
    for p in list_proc:
        p.start()
    for p in list_proc:
        p.join()
    # for debug
    print(' check event: ')
    for i, e in enumerate(list_stop_f):
        print(i, e.is_set())

    while True:
    #check every 5s   
    # cant check is_alive child process of child process
        check_stop(list_proc, list_stop_f)
        
        for p in list_proc:
            print('check PID list_proc:', p,  p.is_alive())
   
        sleep(10)
        # print('check pid after process')
        # tasklist=['python.exe']
        # out=[]
        # for proc in psutil.process_iter():
            # if any(task in proc.name() for task in tasklist):
                # print([{'pid' : proc.pid, 'name' : proc.name()}])

if __name__ == '__main__': 

    freeze_support()
    test()

#Worked!!!!!!!!!!!!!
