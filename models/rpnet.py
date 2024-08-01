import logging

from utils.general import non_max_suppression, xywh2xyxy
from utils.metrics import box_iou

# Define a logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_divisible(x, divisor):
    return int((x + divisor / 2) // divisor * divisor)


import torch
import torch.nn as nn
import torch.nn.functional as F

class LicensePlateClassifier(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, device='cuda',wrPath=None):
        super(LicensePlateClassifier, self).__init__()
        self.device = device
        # self.load_wR2(wrPath)
        self.output_size = prov_num + alpha_num + ad_num
        # self.num_class=num_class
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # ).to(device)

        self.prov_num = prov_num
        self.alpha_num = alpha_num
        self.ad_num = ad_num

        # feature_map_size = input_size // 2
        # flattened_size = 16 * feature_map_size * feature_map_size

        self.classifier1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.prov_num)
        ).to(device)
        self.classifier2 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.alpha_num)
        ).to(device)
        self.classifier3 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier4 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier5 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier6 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier7 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        # self.classifier8 = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, self.ad_num)
        # ).to(device)
    # def load_wR2(self, path):
    #     self.wR2 = wR2(self.num_class)
    #     self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
    #     if not path is None:
    #         self.wR2.load_state_dict(torch.load(path))
    def forward(self, x):
        # x = x.to(self.device)
        # print(f'Input tensor size: {x.size()}')
        # x = self.features(x)
        # x = torch.flatten(x, 1)
        output1 = self.classifier1(x)
        output2 = self.classifier2(x)
        output3 = self.classifier3(x)
        output4 = self.classifier4(x)
        output5 = self.classifier5(x)
        output6 = self.classifier6(x)
        output7 = self.classifier7(x)
        # output8 = self.classifier8(x)
        return [output1, output2, output3, output4, output5, output6, output7]


def adjust_roi(roi, max_height, max_width):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, max_width - 1))
    y1 = max(0, min(y1, max_height - 1))
    x2 = max(x1 + 1, min(x2, max_width))
    y2 = max(y1 + 1, min(y2, max_height))
    return [x1, y1, x2, y2]


import torch
import torch.nn.functional as F

def roi_pooling_ims_yuan(input, rois, size=(7, 7), spatial_scale=1.0):
    # written for one roi one image 从特征图中提取固定大小的特征
    # size: (w, h)
    assert (rois.dim() == 2)
    assert len(input) == len(rois)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)  #batch_size

    rois[:, 1:].mul_(spatial_scale)  #进行缩放，除了第一列
    rois = rois.long()  #转int
    for i in range(num_rois):
        roi = rois[i]
        x1, y1, x2, y2 = roi
        # im = input.narrow(0, im_idx, 1)
        im = input[:, :, y1:y2 + 1, x1:x2 + 1,:]
        # im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1),:]  #取片，将位置特征取出来
        output.append(F.adaptive_max_pool2d(im, size)) #输出的是给定的维度size
        # output.append(torch.nn.AdaptiveAvgPool2d(im, size))

    return torch.cat(output, 0)
def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    assert rois.dim() == 2, f"Expected rois to be 2D, got {rois.dim()}D"
    assert rois.size(1) == 4, f"Expected 4 columns in rois, got {rois.size(1)}"
    assert input.dim() == 5, f"Expected input to be 5D, got {input.dim()}D"

    batch_size, channels, _, _, _ = input.size()
    assert len(input) == len(rois), f"Batch size mismatch: {len(input)} vs {len(rois)}"

    output = []
    rois = rois.data.float()
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()

    for i in range(rois.size(0)):
        roi = rois[i]
        # if roi[0] >= batch_size:
        #     continue  # Skip ROIs for non-existing batches

        x1, y1, x2, y2 = roi
        if x1 < 0 or y1 < 0 or x2 >= input.size(2) or y2 >= input.size(3) or x2 <= x1 or y2 <= y1:
            continue
        # print(f'x1={x1},x2={x2},y1={y1},y2={y2}')
        im = input[:, :, y1:y2 + 1, x1:x2 + 1,:]
        if im.numel() == 0:
            continue
        try:
            pooled = F.adaptive_max_pool2d(im, size)
            output.append(pooled)
        except Exception as e:
            print(f"Error during pooling: {e}, ROI: {roi}")
            continue

    if output:
        return torch.cat(output, 0)
    else:
        return torch.zeros((batch_size, channels, size[0], size[1]), device=input.device)


def roi_pooling_ims1(input, rois, size=(7, 7), spatial_scale=1.0):
    assert rois.dim() == 2
    assert len(input) == len(rois)
    assert rois.size(1) == 4
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()

    assert input.dim() == 5
    batch_size, channels, _, _, _ = input.size()

    for i in range(num_rois):
        roi = rois[i]
        if roi[1] < 0 or roi[2] < 0 or roi[3] >= input.size(2) or roi[4] >= input.size(3) or roi[3] <= roi[1] or roi[4] <= roi[2]:
            continue
        im = input[i:i + 1, :, roi[1]:(roi[3] + 1), roi[2]:(roi[4] + 1)]
        if im.numel() == 0:
            continue
        try:
            pooled = F.adaptive_max_pool2d(im, size)
            output.append(pooled)
        except Exception as e:
            print(f"Error during pooling: {e}, ROI: {roi}")
            continue

    if output:
        return torch.cat(output, 0)
    else:
        return torch.zeros((0, channels, size[0], size[1]), device=input.device)


import torch
import torch.nn as nn


class wR2(nn.Module):
    def __init__(self, flattened_size,num_classes=1000,):
        super(wR2, self).__init__()
        self.device = "cuda"
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        ).to(self.device)

    def forward(self, x):
        x11 = x.view(x.size(0), -1).to(self.device)
        x = self.classifier(x11)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CombinedModel(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, num_class=4, conf_thres=0.25,
                 iou_thres=0.45):
        super(CombinedModel, self).__init__()
        self.prov_num=38
        self.alpha_num=25
        self.ad_num=35
        self.license_plate_classifier = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detection_criterion = nn.MSELoss()  # Example detection loss (e.g., MSELoss)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.wr2 = wR2(num_class).to(self.device)  # Ensure wr2 is on the correct device
        self.wr2 = None
        self.num_class = num_class
    def forward(self, x, detection_targets,YI):
        results = []

        # Check if x is a tuple with two elements
        if len(x) == 2:
            x1, x2, x3 = x[1]  # Assuming x[1] is a tuple of three tensors  # batchsize,
        else:
            x1, x2, x3 = x  # Assuming x is a tuple of three tensors
        # x1=x1.to(dtype=torch.float16)
        # x2=x2.to(dtype=torch.float16)
        # x3=x3.to(dtype=torch.float16)
        batchsize,channel,height,width,time=x3.shape
        flattened_size = channel * height * width * time
        # Pass the third feature map through wR2 to get bounding box locations
        if self.wr2 is None:
            self.wr2 = wR2(flattened_size, self.num_class).to(self.device)  # Forward pass for bounding box prediction
        boxLoc = self.wr2(x3)
        # Extract feature map sizes
        h1, w1 = x1.size(2), x1.size(3)
        h2, w2 = x2.size(2), x2.size(3)
        h3, w3 = x3.size(2), x3.size(3)

        # Define ROI pooling parameters
        p1 = torch.FloatTensor([[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]]).to(self.device)
        p2 = torch.FloatTensor([[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]]).to(self.device)
        p3 = torch.FloatTensor([[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]]).to(self.device)

        # Compute new bounding boxes with postfix adjustments
        postfix = torch.FloatTensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]).to(self.device)
        # postfix = postfix.to(dtype=torch.float16)
        boxNew = boxLoc.matmul(postfix).clamp(min=0, max=1)
        # boxNew = boxNew.to(dtype=torch.float16)  # or torch.float32
        # postfix = postfix.to(dtype=torch.float16)  # or torch.float32

        # Apply ROI pooling to feature maps
        roi1 = roi_pooling_ims(x1, boxNew.matmul(p1), size=(16, 8)).to(self.device)
        roi2 = roi_pooling_ims(x2, boxNew.matmul(p2), size=(16, 8)).to(self.device)
        roi3 = roi_pooling_ims(x3, boxNew.matmul(p3), size=(16, 8)).to(self.device)

        # Concatenate ROIs and pass through classifier
        rois = torch.cat((roi1, roi2, roi3), dim=1).to(self.device)
        _rois = rois.view(rois.size(0), -1).to(self.device)
        _,input_size=_rois.shape
        if self.license_plate_classifier==None:
            self.license_plate_classifier=LicensePlateClassifier(input_size=input_size, prov_num=self.prov_num, alpha_num=self.alpha_num, ad_num=self.ad_num)
        class_preds = self.license_plate_classifier(_rois)
        results.append(class_preds)

        # # Combine results
        # if len(results) > 0:
        #     results_tensor = torch.cat(results, dim=0)
        # else:
        #     results_tensor = torch.zeros((0, 7), device=self.device)

        return results

# class CombinedModel(nn.Module):
#     def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45):
#         super(CombinedModel, self).__init__()
#         self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num)
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#         self.detection_criterion = nn.MSELoss()  # Example detection loss (e.g., MSELoss)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     def forward(self, x, detection_targets):
#         results = []
#         detection_losses = []
#         if len(x)==3:
#             x1 = torch.reshape(x[0], (x[0].shape[0], -1, x[0].shape[-1]))
#             x2 = torch.reshape(x[1], (x[1].shape[0], -1, x[1].shape[-1]))
#             x3 = torch.reshape(x[2], (x[2].shape[0], -1, x[2].shape[-1]))
#             rois = torch.cat((x1, x2, x3), 1)
#             lst = [rois, x]
#             pred = non_max_suppression(lst)
#         else :
#             pred = non_max_suppression(x)
#
#         if not pred or all(p.numel() == 0 for p in pred):
#             return torch.zeros((0, 7), device=self.device)
#
#         for i, roi in enumerate(pred):
#             # Ensure pred is the second class
#             if roi[-1] == 2:  # Assuming the class label is the last element and '2' is the second class
#
#
#                 x1, y1, x2, y2 = map(int, roi[:4])
#                 roi_tensor = x[:, :, y1:y2, x1:x2]  # Crop region of interest
#
#                 # Ensure ROI is large enough to avoid issues
#                 if roi_tensor.size(2) > 0 and roi_tensor.size(3) > 0:
#                     # Calculate detection loss for the current ROI
#                     detection_loss = self.detection_criterion(
#                         torch.tensor([x1, y1, x2, y2], dtype=torch.float32).to(self.device),
#                         detection_targets[i].to(self.device)
#                     )
#                     detection_losses.append(detection_loss.item())
#
#                     if detection_loss.item() > self.conf_thres:
#                         # Proceed with license plate detection if detection loss is above the threshold
#                         roi_height = roi_tensor.size(2) // 7
#                         for j in range(7):
#                             sub_roi = roi_tensor[:, :, j * roi_height:(j + 1) * roi_height, :]
#                             pooled_roi = roi_pooling_ims(
#                                 sub_roi,
#                                 torch.tensor([[0, x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]],
#                                              dtype=torch.float32).to(self.device),
#                                 size=(8, 16)
#                             )  # ROI pooling
#                             class_preds = self.license_plate_classifier(pooled_roi)
#                             results.append(torch.cat((class_preds, torch.tensor(
#                                 [[x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]], device=self.device)), dim=1))
#
#         if len(results) > 0:
#             results_tensor = torch.cat(results, dim=0)
#         else:
#             results_tensor = torch.zeros((0, 7), device=self.device)
#
#         return results_tensor
#
#     def compute_loss(self, detections, targets):
#         detection_loss = 0
#         for detection, target in zip(detections, targets):
#             detection_loss += self.detection_criterion(detection, target)
#         return detection_loss



# class CombinedModel(nn.Module):
#     def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45, device='cuda'):
#         super(CombinedModel, self).__init__()
#         self.device = device
#         self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num).to(device)
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#
#     def iou_loss(self, pred_boxes, target_boxes):
#         x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
#         y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
#         x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
#         y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
#         intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
#
#         pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
#         target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
#
#         union = pred_area + target_area - intersection
#         iou = intersection / union
#         loss = 1 - iou
#         return loss.mean()
#
#     def forward(self, x, detection_targets):
#         results = []
#         detection_losses = []
#
#         # Combine the feature maps from different scales
#         all_boxes = []
#         all_scores = []
#         all_classes = []
#
#         for feature_map in x:
#             batch_size, _, height, width, _ = feature_map.shape
#             box_coords = feature_map[..., :4]
#             confidences = feature_map[..., 4]
#             class_scores = feature_map[..., 5:]
#
#             box_coords = box_coords.reshape(batch_size, -1, 4)
#             confidences = confidences.reshape(batch_size, -1)
#             class_scores = class_scores.reshape(batch_size, -1, class_scores.shape[-1])
#
#             confidences = torch.sigmoid(confidences)
#             class_scores = torch.sigmoid(class_scores)
#
#             box_coords[..., 0] = (box_coords[..., 0] - box_coords[..., 2] / 2) * width
#             box_coords[..., 1] = (box_coords[..., 1] - box_coords[..., 3] / 2) * height
#             box_coords[..., 2] = (box_coords[..., 0] + box_coords[..., 2] / 2) * width
#             box_coords[..., 3] = (box_coords[..., 1] + box_coords[..., 3] / 2) * height
#
#             all_boxes.append(box_coords)
#             all_scores.append(confidences)
#             all_classes.append(class_scores)
#
#         all_boxes = torch.cat(all_boxes, dim=1)
#         all_scores = torch.cat(all_scores, dim=1)
#         all_classes = torch.cat(all_classes, dim=1)
#
#         batch_size = all_boxes.shape[0]
#         for b in range(batch_size):
#             boxes = all_boxes[b]
#             scores = all_scores[b]
#             classes = all_classes[b]
#
#             mask = scores > self.conf_thres
#             boxes = boxes[mask]
#             scores = scores[mask]
#             classes = classes[mask]
#
#             if boxes.size(0) == 0:
#                 continue
#
#             # Concatenate the data for NMS input
#             nms_input = torch.cat((boxes, scores.unsqueeze(1), classes), dim=1)
#
#             # Apply NMS
#             keep = non_max_suppression(nms_input.unsqueeze(0), conf_thres=self.conf_thres, iou_thres=self.iou_thres)
#             boxes = keep[0][:, :4]
#             scores = keep[0][:, 4]
#             classes = keep[0][:, 5]
#             detection_targets=torch.tensor([detection_targets]).to(self.device)
#             for i, box in enumerate(boxes):
#                 x1, y1, x2, y2 = box[:4].int().tolist()
#                 # x1, y1, x2, y2 = map(int, box[:4].detach().cpu().numpy())
#                 if x1 < x2 and y1 < y2:
#                     roi = x[0][:, :, y1:y2, x1:x2]
#
#                     if roi.size(2) > 0 and roi.size(3) > 0:
#                         detection_loss = self.iou_loss(
#                             torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device),
#                             detection_targets
#                         )
#                         detection_losses.append(detection_loss.item())
#
#                         if detection_loss.item() > self.conf_thres:
#                             roi_height = roi.size(2) // 7
#                             character_preds = []
#                             for j in range(7):
#                                 sub_roi = roi[:, :, j * roi_height:(j + 1) * roi_height, :]
#                                 pooled_roi = roi_pooling_ims(
#                                     sub_roi, torch.tensor([[x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]], device=self.device), size=(8, 16)
#                                 )
#             class_preds = self.license_plate_classifier(pooled_roi)
#             character_preds.append(class_preds)
#
#         results.append(torch.cat(character_preds, dim=1))
#
#         if any(results):
#             results_tensor = torch.cat(results, dim=0)
#         else:
#             results_tensor = torch.zeros((0, 7), device=self.device)
#
#         return results_tensor
