# -*- encoding: utf-8 -*-
#Time        :2020/12/13 19:19:15
#Author      :Chen
#FileName    :domain_inversion.py
#Version     :1.0

import _init_paths
import os
import torch
import torch.nn as nn
from utils.load_img import load_img, show_img
import scipy.misc
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.optim as optim
from torch.autograd import Variable
from models.deeplab import Deeplab
from torchvision import transforms
from utils.fda import FDA_source_to_target_np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self):
        self.loss.backward(retain_graph = True)
        return self.loss

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main(image_name):
    model = Deeplab(num_classes=1, pretrained=False, inversion=True).cuda()
    #print(model)
    model.load_state_dict(torch.load('/home/cyang53/CED/Ours/SFDA-TMI/checkpoint/Endo_best.pth'))
    model.apply(fix_bn)
    model.cuda()

    #model.eval()
    criterion = nn.MSELoss()
    content_layers_default = ['conv_4', 'conv_5', 'conv_6']

    #content_img = load_img(os.path.join('./WCE/images', image_name))
    content_img = load_img(os.path.join('/home/cyang53/CED/Data/UDA_Medical/Etislarib/images', image_name))
    content_img = Variable(content_img).cuda()
    wce = model(content_img)
    input_img = content_img.clone()
    # print(ResNet)
    # for name,parameters in ResNet.named_parameters():
    #     print(name,':',parameters.size())

    loss_r_feature_layers = []

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # inputs = torch.randn((1, 3, 256, 256), requires_grad=True, device='cuda',
    #                      dtype=torch.float)

    #optimizer = optim.Adam([inputs], lr=0.2, betas=[0.5, 0.9], eps=1e-8)
    input_param = nn.Parameter(input_img.data)
    #optimizer = optim.LBFGS([input_param])
    optimizer = optim.Adam([input_param], lr=0.01, betas=[0.5, 0.9], eps=1e-8)

    for i in range(100):
        optimizer.zero_grad()
        output = model(input_param)
        loss_content = criterion(output[1], wce[1]) + criterion(output[2], wce[2]) + criterion(output[0], wce[0])
        #print(len(loss_r_feature_layers))
        loss_r_feature = sum([mod.r_feature for (idx, mod) in
                              enumerate(loss_r_feature_layers[1:24])])
        loss = loss_r_feature + 0.5 * loss_content
        loss.backward(retain_graph=True)
        optimizer.step()
        #print(loss_r_feature, loss_content)
    input_param.data.clamp_(0, 1)
    #show_img(input_param.data.cpu())
    save_pic = transforms.ToPILImage()(input_param.data.cpu().squeeze(0))

    save_pic.save('/home/cyang53/CED/Data/UDA_Medical/Etislarib/coarse_generation/' + image_name)
    #save_pic.save('./picture/generate.png')

# ResNet = models.resnet50(pretrained=True)
# ResNet = ResNet.cuda()
# for module in ResNet.modules():
#     print(module)
if __name__ == '__main__':
    target_dir = '/home/cyang53/CED/Data/UDA_Medical/Etislarib/images'
    source_dir = '/home/cyang53/CED/Data/UDA_Medical/Etislarib/coarse_generation'
    dir_list = os.listdir(target_dir)
    for i in dir_list:
        #if i != '83.png':
        main(i)
        im_src = Image.open(os.path.join(target_dir, i)).convert('RGB')
        im_trg = Image.open(os.path.join(source_dir, i)).convert('RGB')

        im_src = im_src.resize((256, 256), Image.BICUBIC)
        im_trg = im_trg.resize((256, 256), Image.BICUBIC)

        im_src = np.asarray(im_src, np.float32)
        im_trg = np.asarray(im_trg, np.float32)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))

        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.05)

        src_in_trg = src_in_trg.transpose((1, 2, 0))
        scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(os.path.join('/home/cyang53/CED/Data/UDA_Medical/Etislarib/fine_generation', i))
            #main('./picture/WCE.png')
