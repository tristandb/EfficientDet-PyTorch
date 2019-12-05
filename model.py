import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
import numpy as np
import geffnet
from tqdm import tqdm 

feature_sizes = [64, 88, 112, 160, 224, 288, 384, 384]
geffnets = [geffnet.tf_efficientnet_b0, geffnet.tf_efficientnet_b1, geffnet.tf_efficientnet_b2, geffnet.tf_efficientnet_b3, 
            geffnet.tf_efficientnet_b4, geffnet.tf_efficientnet_b5, geffnet.tf_efficientnet_b6, geffnet.tf_efficientnet_b7]

def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

class PyramidFeatures(nn.Module):
    def __init__(self, size, feature_size=256, epsilon=0.0001, index=0):
        super(PyramidFeatures, self).__init__()
        self.epsilon = epsilon
        self.index = index
        
        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        
        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)
        
        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        )
        
        self.p3_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p4_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p5_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p6_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p7_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        
        self.p3_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p4_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p5_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p6_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        self.p7_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1), nn.BatchNorm2d(feature_size, momentum=0.9997, eps=4e-5))
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3))
        self.w2_relu = nn.ReLU()
        

    def forward(self, inputs):
        if self.index == 0:
            c3, c4, c5 = inputs
            # Calculate the input column of BiFPN
            p3_x = self.p3(c3)        
            p4_x = self.p4(c4)
            p5_x = self.p5(c5)
            p6_x = self.p6(c5)
            p7_x = self.p7(p6_x)
        else:
            p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0] * p6_x + w1[1] * F.interpolate(p7_x, scale_factor=2, mode='nearest'))        
        p5_td = self.p5_td(w1[0] * p5_x + w1[1] * F.interpolate(p6_x, scale_factor=2, mode='nearest'))
        p4_td = self.p4_td(w1[0] * p4_x + w1[1] * F.interpolate(p5_x, scale_factor=2, mode='nearest'))
        p3_td = self.p3_td(w1[0] * p3_x + w1[1] * F.interpolate(p4_x, scale_factor=2, mode='nearest'))
        
        # Calculate Bottom-Up Pathway
        p7_out = self.p7_out(w2[0] * p7_x + w2[1] * p7_td + w2[2] * F.interpolate(p6_td, scale_factor=0.5, mode='nearest'))
        p6_out = self.p6_out(w2[0] * p6_x + w2[1] * p6_td + w2[2] * F.interpolate(p5_td, scale_factor=0.5, mode='nearest'))
        p5_out = self.p5_out(w2[0] * p5_x + w2[1] * p5_td + w2[2] * F.interpolate(p4_td, scale_factor=0.5, mode='nearest'))
        p4_out = self.p4_out(w2[0] * p4_x + w2[1] * p4_td + w2[2] * F.interpolate(p3_td, scale_factor=0.5, mode='nearest'))
        p3_out = p3_td

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class RegressionModel(nn.Module):
    def __init__(self, phi, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        num_class_layers = 3 + phi//3
        
        modules = []
        
        for i in range(num_class_layers):
            modules.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1))
            modules.append(nn.ReLU())
        
        self.net = nn.Sequential(*modules)
        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.net(x)
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, phi, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        num_class_layers = 3 + phi//3
        
        modules = []
        
        for i in range(num_class_layers):
            modules.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1))
            modules.append(nn.ReLU())
        
        self.net = nn.Sequential(*modules)

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.net(x)
        
        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
    
class EfficientDet(nn.Module):
    def __init__(self, num_classes, phi):
        feature_size = feature_sizes[phi]
        super(EfficientDet, self).__init__()
        
        self.backbone = geffnets[phi](pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
        
        # Get backbone feature sizes. 
        fpn_sizes = [40, 80, 192]
        
        self.fpn = [PyramidFeatures(fpn_sizes, feature_size=feature_size, index=index).cuda() for index in range(min(2+phi, 8))]
        
        self.regressionModel = RegressionModel(phi, feature_size=feature_size)
        self.classificationModel = ClassificationModel(phi, feature_size=feature_size, num_classes=num_classes)
        
        self.anchors = Anchors()
        
        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.focalLoss = losses.FocalLoss()
        
        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        
        x = self.backbone.conv_stem(img_batch)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        
        features = []
        for block in self.backbone.blocks:
            x = block(x)   
            if block[0].conv_dw.stride == (2, 2):
                features.append(x)
        features = features[1:]
                
        for fpn_block in self.fpn:
            features = fpn_block(features)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

def efficientdet(num_classes, pretrained=False, phi=0,**kwargs):
    """Constructs a EfficientDet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        phi (int)        : Scaling coefficient.
    """
    model = EfficientDet(num_classes, phi, **kwargs)
    return model
