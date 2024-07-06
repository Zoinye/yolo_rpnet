import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(x, divisor):
    return int((x + divisor / 2) // divisor * divisor)


class LicensePlateClassifier(nn.Module):
    def __init__(self, input_size, prov_num=38, alpha_num=25, ad_num=35):
        super(LicensePlateClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算卷积层输出的特征图大小
        feature_map_size = input_size // 2  # 因为有一个2x2的池化层

        self.classifier1 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, prov_num)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, alpha_num)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
        )
        self.classifier8 = nn.Sequential(
            nn.Linear(16 * feature_map_size * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ad_num)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 测试 LicensePlateClassifier
input_size = 256
model = LicensePlateClassifier(input_size).to(device)
input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
outputs = model(input_tensor)
for i, output in enumerate(outputs):
    print(f"Output {i + 1} shape: {output.shape}")
