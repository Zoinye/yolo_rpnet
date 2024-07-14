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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import logging

from utils.general import non_max_suppression

# Define a logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_divisible(x, divisor):
    return int((x + divisor / 2) // divisor * divisor)


class LicensePlateClassifier(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35):
        super(LicensePlateClassifier, self).__init__()
        self.output_size = prov_num + alpha_num + ad_num  # 假设这是你的输出尺寸
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.prov_num = prov_num
        self.alpha_num = alpha_num
        self.ad_num = ad_num
        # 计算卷积层输出的特征图大小
        feature_map_size = input_size // 2  # 因为有一个2x2的池化层
        flattened_size = 16 * feature_map_size * feature_map_size

        self.classifier1 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.prov_num)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.alpha_num)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )
        self.classifier8 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.ad_num)
        )

    def forward(self, x):
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


def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    assert (rois.dim() == 2)
    assert len(input) == len(rois)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)  # batch_size

    rois[:, 1:].mul_(spatial_scale)  # 进行缩放，除了第一列
    rois = rois.long()  # 转int
    for i in range(num_rois):
        roi = rois[i]
        im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]  # 取片，将位置特征取出来
        output.append(F.adaptive_max_pool2d(im, size))  # 输出的是给定的维度size
    return torch.cat(output, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedModel(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45, device='cuda'):
        super(CombinedModel, self).__init__()
        self.device = device
        self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num).to(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detection_criterion = nn.MSELoss()  # Example detection loss (e.g., MSELoss)

    def forward(self, x, detection_targets):
        results = []
        detection_losses = []

        pred = non_max_suppression(x)

        for i, roi in enumerate(pred):
            if roi is None or len(roi) < 4:
                continue  # 跳过无效的或不完整的ROI

            x1, y1, x2, y2 = map(int, roi[:4])
            roi = x[:, :, y1:y2, x1:x2]  # Crop region of interest

            # Ensure ROI is large enough to avoid issues
            if roi.size(2) > 0 and roi.size(3) > 0:
                # Calculate detection loss for the current ROI
                detection_loss = self.detection_criterion(
                    torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=self.device),
                    detection_targets[i].to(self.device)
                )
                detection_losses.append(detection_loss.item())

                if detection_loss.item() > self.conf_thres:
                    # Proceed with license plate detection if detection loss is above the threshold
                    roi_height = roi.size(2) // 7
                    character_preds = []
                    for j in range(7):
                        sub_roi = roi[:, :, j * roi_height:(j + 1) * roi_height, :]
                        pooled_roi = roi_pooling_ims(
                            sub_roi, torch.tensor([[0, x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]], device=self.device), size=(8, 16)
                        )  # ROI pooling
                        class_preds = self.license_plate_classifier(pooled_roi)
                        character_preds.append(class_preds)

                    # 将每个字符的预测结果拼接成一个单独的张量
                    results.append(torch.cat(character_preds, dim=1))

        # 将所有结果拼接成最终的张量
        # if results:
        #     results_tensor = torch.cat(results, dim=0)
        # else:
        #     results_tensor = torch.zeros((0, 7), device=self.device)
        if any(results):  # 检查 results 中是否有非空的元素
            results_tensor = torch.cat(results, dim=0)
        else:
            results_tensor = torch.zeros((0, 7), device=self.device)

        return results_tensor




