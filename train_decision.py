################################################两阶段网络决策网络的训练####################################################
from models import SegmentNet, DecisionNet, weights_init_normal
from dataset import KolektorDataset
import numpy as np

import torch.nn as nn
import torch

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import argparse
import time
import PIL.Image as Image

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------设置参数--------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")
parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=61, help="end_epoch")
parser.add_argument("--seg_epoch", type=int, default=50, help="pretrained segment epoch")

parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")


parser.add_argument("--img_height", type=int, default=704, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")


opt = parser.parse_args()

print(opt)

dataSetRoot = "./Data" # "/home/sean/Data/KolektorSDD_sean"

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------构建网络-------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

segment_net = SegmentNet(init_weights=True)
decision_net = DecisionNet(init_weights=True)

criterion_decision = torch.nn.MSELoss() #损失函数的设置

if opt.cuda:
    segment_net = segment_net.cuda()
    decision_net = decision_net.cuda()
    #criterion_segment.cuda()
    criterion_decision.cuda()

if opt.gpu_num > 1:
    segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))
    decision_net = torch.nn.DataParallel(decision_net, device_ids=list(range(opt.gpu_num)))

if opt.begin_epoch != 0:
    # Load pretrained models
    decision_net.load_state_dict(torch.load("./saved_models/decision_net_%d.pth" % (opt.begin_epoch)))
else:
    # Initialize weights
    decision_net.apply(weights_init_normal)


segment_net.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.seg_epoch))) # 加载预训练好的分割模型（储存在训练分割模型时建立的saved_models文件夹中）

optimizer_dec = torch.optim.Adam(decision_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) #优化器设置

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------图像预处理-----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height//8, opt.img_width//8)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------图像加载-----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
trainOKloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask, subFold="Train_OK", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)

trainNGloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_,  transforms_mask= transforms_mask, subFold="Train_NG", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)

testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask,  subFold="Test", isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------正式训练-----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    for epoch in range(opt.begin_epoch, opt.end_epoch):

        iterOK = trainOKloader.__iter__() # 这里iterOK是一个multiProcessingDataLoaderIter：大小为iterOK文件夹中图片的数量/batchsize
                                        #这样的访问方式返回的iterOK是一个基本的迭代器，用于一会一个batch一个batch地提取数据进行训练
        iterNG = trainNGloader.__iter__()

        lenNum = min( len(trainNGloader), len(trainOKloader))
        lenNum = 2*(lenNum-1)

        # ---------------------------------挑选第i个batch(一个epoch中共lenNum个batch)的图片进行训练---------------------------
        for i in range(0, lenNum):

            if i % 2 == 0:
                batchData = iterOK.__next__()
                gt_c = Variable(torch.Tensor(np.zeros((batchData["img"].size(0), 1))), requires_grad=False)#产生和batchData中一样大小的全0的tensor

                '''
                #显示batchData中第一张图片
                a = batchData['img'][0]
                a = transforms.ToPILImage()(a)
                a.show()
                '''

            else :
                batchData = iterNG.__next__()
                gt_c = Variable(torch.Tensor(np.ones((batchData["img"].size(0), 1))), requires_grad=False)


            if opt.cuda:
                img = batchData["img"].cuda()
                mask = batchData["mask"].cuda()
                gt_c = gt_c.cuda()
            else:
                img = batchData["img"]
                mask = batchData["mask"]

            rst = segment_net(img) #分割网络对batchData的输出结果

            f = rst["f"] #分割网络的第4层输出
            seg = rst["seg"] #分割网络的第5层/最后一层输出

            optimizer_dec.zero_grad()

            rst_d = decision_net(f, seg) #决策网络的输出
            # rst_d = torch.Tensor.long(rst_d)

            loss_dec = criterion_decision(rst_d, gt_c) #决策网络输出和全0数据做损失函数，决策网络的输出应该越小越好

            loss_dec.backward()
            optimizer_dec.step()

            sys.stdout.write(
                "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
                 %(
                    epoch,
                    opt.end_epoch,
                    i,
                    lenNum,
                    loss_dec.item()
                 )
            )

        # -----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------验证部分-------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------

        if opt.need_test and epoch % opt.test_interval == 0 and epoch >= opt.test_interval:

            for i, testBatch in enumerate(testloader):
                imgTest = testBatch["img"].cuda()
                rstTest = segment_net(imgTest)

                fTest = rstTest["f"]
                segTest = rstTest["seg"]

                cTest = decision_net(fTest, segTest)

                save_path_str = "./testResultDec/epoch_%d"%epoch
                if os.path.exists(save_path_str) == False:
                    os.makedirs(save_path_str, exist_ok=True)

                if cTest.item() > 0.5:
                    labelStr = "NG"
                else:
                    labelStr = "OK"

                save_image(imgTest.data, "%s/img_%d_%s.jpg"% (save_path_str, i , labelStr))
                save_image(segTest.data, "%s/img_%d_seg_%s.jpg"% (save_path_str, i, labelStr))

        # -----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------储存网络-------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------

        if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:

            save_path_str = "./saved_models"
            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)

            torch.save(decision_net.state_dict(), "%s/decision_net_%d.pth" % (save_path_str, epoch))
            print("save weights ! epoch = %d"%epoch)
            pass
