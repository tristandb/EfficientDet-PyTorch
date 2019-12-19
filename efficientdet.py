import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
from torchvision.ops import nms

from efficientnet_pytorch import EfficientNet

from bifpn import BiFPN

from timeitdec import timeit

w_bifpn = [64, 88, 112, 160, 224, 288, 384, 384]

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, d_class=3, num_anchors=9, feature_size=64):
        super(RegressionModel, self).__init__()
        
        prediction_net = []
        for _ in range(d_class):
            prediction_net.append(nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1))
            prediction_net.append(nn.ReLU())
            num_features_in = feature_size
        self.prediction_net = nn.Sequential(*prediction_net)

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.prediction_net(x)
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, d_class=3, num_classes=80, prior=0.01, feature_size=64):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        classification_net = []
        for _ in range(d_class):
            classification_net.append(nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1))
            classification_net.append(nn.ReLU())
            num_features_in = feature_size
        self.classification_net = nn.Sequential(*classification_net)

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.classification_net(x)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
    
class EfficientDet(nn.Module):

    def __init__(self, num_classes, block, pretrained=False, phi=0):
        self.inplanes = w_bifpn[phi]
        super(EfficientDet, self).__init__()
        efficientnet = EfficientNet.from_pretrained(f'efficientnet-b{phi}')
        blocks = []
        count = 0
        fpn_sizes = []
        for block in efficientnet._blocks:
            blocks.append(block)
            if block._depthwise_conv.stride == [2, 2]:
                count += 1
                fpn_sizes.append(block._project_conv.out_channels)
                if len(fpn_sizes) >= 4:
                    break
                    
        self.efficientnet = nn.Sequential(efficientnet._conv_stem, efficientnet._bn0, *blocks)
        num_layers = min(phi+2, 8)
        self.fpn = BiFPN(fpn_sizes[1:], feature_size=w_bifpn[phi], num_layers=num_layers)
        
        d_class = 3 + (phi // 3)
        self.regressionModel = RegressionModel(w_bifpn[phi], feature_size=w_bifpn[phi], d_class=d_class)
        self.classificationModel = ClassificationModel(w_bifpn[phi], feature_size=w_bifpn[phi], d_class=d_class, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.focalLoss = losses.FocalLoss().cuda()
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
                    
        x = self.efficientnet[0](img_batch)
        x = self.efficientnet[1](x)
        
        # Forward batch trough backbone
        features = []
        for block in self.efficientnet[2:]:
            x = block(x)   
            if block._depthwise_conv.stride == [2, 2]:
                features.append(x)

        features = self.fpn(features[1:])
        
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

            anchors_nms_idx = nms(transformed_anchors, scores, 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
        

def efficientdet(num_classes, pretrained=True, **kwargs):
    """Constructs an EfficientDet
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
    """
    model = EfficientDet(num_classes, Bottleneck, pretrained=pretrained, **kwargs)
    return model