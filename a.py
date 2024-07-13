import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import logging

# Define a logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_divisible(x, divisor):
    return int((x + divisor / 2) // divisor * divisor)


class LicensePlateClassifier(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35):
        super(LicensePlateClassifier, self).__init__()

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


class CombinedModel(nn.Module):
    def __init__(self, input_size=640, prov_num=38, alpha_num=25, ad_num=35, conf_thres=0.25, iou_thres=0.45):
        super(CombinedModel, self).__init__()
        self.license_plate_classifier = LicensePlateClassifier(input_size, prov_num, alpha_num, ad_num)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detection_criterion = nn.CrossEntropyLoss()
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, x, rois, detection_targets, classification_targets):
        results = []
        classification_losses = []

        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = map(int, roi)
            roi = x[:, :, y1:y2, x1:x2]  # Crop region of interest

            # Ensure ROI is large enough to avoid issues
            if roi.size(2) > 0 and roi.size(3) > 0:
                # 将区域分成7份
                roi_height = roi.size(2) // 7
                for j in range(7):
                    sub_roi = roi[:, :, j * roi_height:(j + 1) * roi_height, :]
                    pooled_roi = roi_pooling_ims(sub_roi, torch.tensor([[0, x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height]]), size=(8, 16))  # ROI pooling
                    class_preds = self.license_plate_classifier(pooled_roi)

                    # Calculate detection loss for the current sub-ROI
                    detection_loss = self.detection_criterion(class_preds[0], detection_targets[i])
                    if detection_loss < self.conf_thres:
                        # If the detection loss is below the threshold, consider it as a license plate and compute classification loss
                        classification_loss = sum(self.classification_criterion(p, t) for p, t in zip(class_preds, classification_targets[i]))
                        classification_losses.append(classification_loss)
                        results.append((class_preds, (x1, y1 + j * roi_height, x2, y1 + (j + 1) * roi_height)))

        total_classification_loss = sum(classification_losses)
        return results, total_classification_loss

    def compute_detection_loss(self, detections, targets):
        detection_loss = 0
        for detection, target in zip(detections, targets):
            cls_pred = detection[:, -1]  # Get predicted class scores
            target_cls = target[-1]  # Get target class
            detection_loss += self.detection_criterion(cls_pred, target_cls)
        return detection_loss

    def compute_classification_loss(self, predictions, targets):
        total_loss = 0
        for pred, target in zip(predictions, targets):
            loss = sum(self.classification_criterion(p, t) for p, t in zip(pred, target))
            total_loss += loss
        return total_loss

    def compute_loss(self, detections, detection_targets, classification_preds, classification_targets):
        detection_loss = self.compute_detection_loss(detections, detection_targets)
        classification_loss = self.compute_classification_loss(classification_preds, classification_targets)
        return detection_loss + classification_loss


if __name__ == '__main__':
    # Create a sample input
    input = ag.Variable(torch.rand(2, 3, 640, 640), requires_grad=True)  # Example input
    rois = ag.Variable(torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]]), requires_grad=False)  # Example ROIs

    model = CombinedModel()
    preds, classification_loss = model(input, rois, torch.tensor([1, 2]), [[torch.tensor([1]), torch.tensor([2]), torch.tensor([3, 4, 5])]])

    # Example targets for loss computation (you should replace this with actual target data)
    detection_targets = [torch.randint(0, 3, (1,)) for _ in range(len(rois))]
    classification_targets = [[torch.randint(0, 38, (1,)), torch.randint(0, 25, (1,)), torch.randint(0, 35, (5,))] for _ in range(len(rois))]

    # Compute loss
    loss = model.compute_loss(detection_targets, detection_targets, preds, classification_targets)
    loss.backward()

    print(f"Loss: {loss.item()}")
