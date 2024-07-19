# import torch
# import torch.nn as nn
# import contextlib
# import logging
#
# from utils.general import non_max_suppression
#
# # Define a logger
# LOGGER = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
#
# def make_divisible(x, divisor):
#     return int((x + divisor / 2) // divisor * divisor)
#
#
# class LicensePlateClassifier(nn.Module):
#     def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35):
#         super(LicensePlateClassifier, self).__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.prov_num=prov_num
#         self.alpha_num=alpha_num
#         self.ad_num=ad_num
#         # 计算卷积层输出的特征图大小
#         feature_map_size = input_size // 2  # 因为有一个2x2的池化层
#         flattened_size = 16 * feature_map_size * feature_map_size
#
#         self.classifier1 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.prov_num)
#         )
#         self.classifier2 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.alpha_num)
#         )
#         self.classifier3 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#         self.classifier4 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#         self.classifier5 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#         self.classifier6 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#         self.classifier7 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#         self.classifier8 = nn.Sequential(
#             nn.Linear(flattened_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.ad_num)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         output1 = self.classifier1(x)
#         output2 = self.classifier2(x)
#         output3 = self.classifier3(x)
#         output4 = self.classifier4(x)
#         output5 = self.classifier5(x)
#         output6 = self.classifier6(x)
#         output7 = self.classifier7(x)
#         output8 = self.classifier8(x)
#         return [output1, output2, output3, output4, output5, output6, output7, output8]
#
#
# class CombinedModel(nn.Module):
#     def __init__(self,  input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45):
#         super(CombinedModel, self).__init__()
#         # self.detect_model = detect_model
#         self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num)
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#
#     def forward(self, x):
#         # Get YOLO detections
#         # detect_output = self.detect_model(x)
#         # pred, _ = detect_output
#         # Apply non-max suppression (NMS) to filter detections
#         detections = non_max_suppression(x, self.conf_thres, self.iou_thres)
#         # Container for results
#         results = []
#         # Iterate over each detection and pass through license_plate_classifier
#         for det in detections:
#             if det is not None and len(det):
#                 for *xyxy, conf, cls in det:
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     roi = x[:, :, y1:y2, x1:x2]  # Crop region of interest
#
#                     # Ensure ROI is large enough to avoid issues
#                     if roi.size(2) > 0 and roi.size(3) > 0:
#                         class_preds = self.license_plate_classifier(roi)
#                         results.append((class_preds, conf, cls, (x1, y1, x2, y2)))
#         return results
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
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
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, device='cuda'):
        super(LicensePlateClassifier, self).__init__()
        self.device = device
        self.output_size = prov_num + alpha_num + ad_num
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to(device)

        self.prov_num = prov_num
        self.alpha_num = alpha_num
        self.ad_num = ad_num

        feature_map_size = input_size // 2
        flattened_size = 16 * feature_map_size * feature_map_size

        self.classifier1 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.prov_num)
        ).to(device)
        self.classifier2 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.alpha_num)
        ).to(device)
        self.classifier3 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier4 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier5 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier6 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier7 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)
        self.classifier8 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        ).to(device)

    def forward(self, x):
        # x = x.to(self.device)
        # print(f'Input tensor size: {x.size()}')
        x = self.features(x)
        x = torch.flatten(x, 1)
        output1 = self.classifier1(x)
        output2 = self.classifier2(x)
        output3 = self.classifier3(x)
        output4 = self.classifier4(x)
        output5 = self.classifier5(x)
        output6 = self.classifier6(x)
        output7 = self.classifier7(x)
        output8 = self.classifier8(x)
        return [output1, output2, output3, output4, output5, output6, output7, output8]






def adjust_roi(roi, max_height, max_width):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, max_width - 1))
    y1 = max(0, min(y1, max_height - 1))
    x2 = max(x1 + 1, min(x2, max_width))
    y2 = max(y1 + 1, min(y2, max_height))
    return [x1, y1, x2, y2]
def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
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

# def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
#     assert rois.dim() == 2
#     assert len(input) == len(rois)
#     assert rois.size(1) == 4  # Including batch index
#     output = []
#     rois = rois.data.float()
#     num_rois = rois.size(0)
#
#     rois[:, 1:].mul_(spatial_scale)
#     rois = rois.long()
#
#     # Ensure input tensor is 5-dimensional
#     assert input.dim() == 5
#     batch_size, channels, depth, height, width = input.size()
#
#     for i in range(num_rois):
#         roi = rois[i]
#         batch_ind = roi[0]
#         x1, y1, x2, y2 = adjust_roi(roi[:], height, width)
#
#         if x2 > x1 and y2 > y1:  # Ensure valid ROI
#             im = input[batch_ind:batch_ind + 1, :, :, y1:(y2 + 1), x1:(x2 + 1)]
#             if im.size(3) > 0 and im.size(4) > 0:  # Ensure non-empty spatial dimensions
#                 try:
#                     pooled = F.adaptive_max_pool2d(im, size)
#                     output.append(pooled)
#                 except Exception as e:
#                     print(f"Error during pooling: {e}, ROI: {roi}, im size: {im.size()}")
#                     continue
#
#     if output:
#         return torch.cat(output, 0)
#     else:
#         return torch.zeros((0, channels, *size), device=input.device)  # Return empty tensor with correct dimensions





import torch
import torch.nn as nn
import torchvision

class CombinedModel(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45, device='cuda'):
        super(CombinedModel, self).__init__()
        self.device = device
        self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num).to(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def iou_loss(self, pred_boxes, target_boxes):
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

        union = pred_area + target_area - intersection
        iou = intersection / union
        loss = 1 - iou
        return loss.mean()

    def forward(self, x, detection_targets):
        results = []
        detection_losses = []

        # Combine the feature maps from different scales
        all_boxes = []
        all_scores = []
        all_classes = []

        for feature_map in x:
            batch_size, _, height, width, _ = feature_map.shape
            box_coords = feature_map[..., :4]
            confidences = feature_map[..., 4]
            class_scores = feature_map[..., 5:]

            box_coords = box_coords.reshape(batch_size, -1, 4)
            confidences = confidences.reshape(batch_size, -1)
            class_scores = class_scores.reshape(batch_size, -1, class_scores.shape[-1])

            confidences = torch.sigmoid(confidences)
            class_scores = torch.sigmoid(class_scores)

            box_coords[..., 0] = (box_coords[..., 0] - box_coords[..., 2] / 2) * width
            box_coords[..., 1] = (box_coords[..., 1] - box_coords[..., 3] / 2) * height
            box_coords[..., 2] = (box_coords[..., 0] + box_coords[..., 2] / 2) * width
            box_coords[..., 3] = (box_coords[..., 1] + box_coords[..., 3] / 2) * height

            all_boxes.append(box_coords)
            all_scores.append(confidences)
            all_classes.append(class_scores)

        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_classes = torch.cat(all_classes, dim=1)

        batch_size = all_boxes.shape[0]
        for b in range(batch_size):
            boxes = all_boxes[b]
            scores = all_scores[b]
            classes = all_classes[b]

            mask = scores > self.conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            if boxes.size(0) == 0:
                continue

            # Concatenate the data for NMS input
            nms_input = torch.cat((boxes, scores.unsqueeze(1), classes), dim=1)

            # Apply NMS
            keep = non_max_suppression(nms_input.unsqueeze(0), conf_thres=self.conf_thres, iou_thres=self.iou_thres)
            boxes = keep[0][:, :4]
            scores = keep[0][:, 4]
            classes = keep[0][:, 5]
            detection_targets=torch.tensor([detection_targets]).to(self.device)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4].int().tolist()
                # x1, y1, x2, y2 = map(int, box[:4].detach().cpu().numpy())
                if x1 < x2 and y1 < y2:
                    roi = x[0][:, :, y1:y2, x1:x2]

                    if roi.size(2) > 0 and roi.size(3) > 0:
                        detection_loss = self.iou_loss(
                            torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device),
                            detection_targets
                        )
                        detection_losses.append(detection_loss.item())

                        if detection_loss.item() > self.conf_thres:
                            roi_height = roi.size(2) // 7
                            character_preds = []
                            for j in range(7):
                                sub_roi = roi[:, :, j * roi_height:(j + 1) * roi_height, :]
                                pooled_roi = roi_pooling_ims(
                                    sub_roi, torch.tensor([[x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]], device=self.device), size=(8, 16)
                                )
                                class_preds = self.license_plate_classifier(pooled_roi)
                                character_preds.append(class_preds)

                            results.append(torch.cat(character_preds, dim=1))

        if any(results):
            results_tensor = torch.cat(results, dim=0)
        else:
            results_tensor = torch.zeros((0, 7), device=self.device)

        return results_tensor

# def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nm=0):
#     assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
#     assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
#     if isinstance(prediction, (list, tuple)):
#         prediction = prediction[0]
#
#     device = prediction.device
#     mps = "mps" in device.type
#     if mps:
#         prediction = prediction.cpu()
#     bs = prediction.shape[0]
#     nc = prediction.shape[2] - nm - 5
#     xc = prediction[..., 4] > conf_thres
#
#     max_wh = 7680
#     max_nms = 30000
#     time_limit = 0.5 + 0.05 * bs
#     redundant = True
#     multi_label &= nc > 1
#     merge = False
#
#     t = time.time()
#     mi = 5 + nc
#     output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     for xi, x in enumerate(prediction):
#         x = x[xc[xi]]
#
#         if labels and len(labels[xi]):
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#             v[:, :4] = lb[:, 1:5]
#             v[:, 4] = 1.0
#             v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
#             x = torch.cat((x, v), 0)
#
#         if not x.shape[0]:
#             continue
#
#         x[:, 5:] *= x[:, 4:5]
#
#         box = xywh2xyxy(x[:, :4])
#         mask = x[:, mi:]
#
#         if multi_label:
#             i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
#         else:
#             conf, j = x[:, 5:mi].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
#
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#         n = x.shape[0]
#         if not n:
#             continue
#         x = x[x[:, 4].argsort(descending=True)[:max_nms]]
#
#         c = x[:, 5:6] * (0 if agnostic else max_wh)
#         boxes, scores = x[:, :4] + c, x[:, 4]
#         i = torchvision.ops.nms(boxes, scores, iou_thres)
#         i = i[:max_det]
#         if merge and (1 < n < 3e3):
#             iou = box_iou(boxes[i], boxes) > iou
#





# def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
#     assert (rois.dim() == 2)
#     assert len(input) == len(rois)
#     assert (rois.size(1) == 5)
#     output = []
#     rois = rois.data.float()
#     num_rois = rois.size(0)  # batch_size
#
#     rois[:, 1:].mul_(spatial_scale)  # 进行缩放，除了第一列
#     rois = rois.long()  # 转int
#     for i in range(num_rois):   #1,3,4,13,8
#         roi = rois[i]
#         im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]  # 取片，将位置特征取出来
#         output.append(F.adaptive_max_pool2d(im, size))  # 输出的是给定的维度size
#     return torch.cat(output, 0)
#
#
# import torch
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CombinedModel(nn.Module):
#     def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45, device='cuda'):
#         super(CombinedModel, self).__init__()
#         self.device = device
#         self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num).to(device)
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#         self.detection_criterion = nn.MSELoss()  # Example detection loss (e.g., MSELoss)
#
#     def forward(self, x, detection_targets):
#         results = []
#         detection_losses = []
#
#         pred = non_max_suppression(x)
#
#         for i, roi in enumerate(pred):
#             if roi is None or len(roi) < 4:
#                 continue  # 跳过无效的或不完整的ROI
#
#             x1, y1, x2, y2 = map(int, roi[:4])
#             roi = x[:, :, y1:y2, x1:x2]  # Crop region of interest
#
#             # Ensure ROI is large enough to avoid issues
#             if roi.size(2) > 0 and roi.size(3) > 0:
#                 # Calculate detection loss for the current ROI
#                 detection_loss = self.detection_criterion(
#                     torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=self.device),
#                     detection_targets[i].to(self.device)
#                 )
#                 detection_losses.append(detection_loss.item())
#
#                 if detection_loss.item() > self.conf_thres:
#                     # Proceed with license plate detection if detection loss is above the threshold
#                     roi_height = roi.size(2) // 7
#                     character_preds = []
#                     for j in range(7):
#                         sub_roi = roi[:, :, j * roi_height:(j + 1) * roi_height, :]
#                         pooled_roi = roi_pooling_ims(
#                             sub_roi, torch.tensor([[0, x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]], device=self.device), size=(8, 16)
#                         )  # ROI pooling
#                         class_preds = self.license_plate_classifier(pooled_roi)
#                         character_preds.append(class_preds)
#
#                     # 将每个字符的预测结果拼接成一个单独的张量
#                     results.append(torch.cat(character_preds, dim=1))
#
#         # 将所有结果拼接成最终的张量
#         # if results:
#         #     results_tensor = torch.cat(results, dim=0)
#         # else:
#         #     results_tensor = torch.zeros((0, 7), device=self.device)
#         if any(results):  # 检查 results 中是否有非空的元素
#             results_tensor = torch.cat(results, dim=0)
#         else:
#             results_tensor = torch.zeros((0, 7), device=self.device)
#
#         return results_tensor