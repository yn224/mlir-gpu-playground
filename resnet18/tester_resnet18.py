from PIL import Image
import requests

import torch
from torchvision import transforms

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from layer_specific.batchnorm2d import BN1

def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)

def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels

def top3_possibilities(res, labels):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3

def predictions(torch_func, jit_func, img, labels):
    golden_prediction = top3_possibilities(torch_func(img), labels)
    print("==================")
    print("PyTorch prediction")
    print(golden_prediction)

    prediction = top3_possibilities(torch.from_numpy(jit_func(img.numpy())), labels)
    print("==================")
    print("torch-mlir prediction")
    print(prediction)

##########################################################################
# Model Definition
##########################################################################
# References for manual implementation
# Identity downsampling omitted
# https://d2l.ai/chapter_convolutional-modern/batch-norm.html
# https://niko-gamulin.medium.com/resnet-implementation-with-pytorch-from-scratch-23cf3047cb93
# https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
# https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch

class ModResnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN1(self.inplanes, 4) # torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #####
        # Layer1 - self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer1_conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer1_bn1 = BN1(64, 4)
        self.layer1_conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer1_bn2 = BN1(64, 4)
        # self.layer1_iden_conv = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1)
        # self.layer1_iden_bn = BN1(64, 4)
        #####

        #####
        # Layer2 - self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer2_conv1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer2_bn1 = BN1(128, 4)
        self.layer2_conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.layer2_bn2 = BN1(128, 4)
        # self.layer2_iden_conv = torch.nn.Conv2d(64, 128, kernel_size=1, stride=2)
        # self.layer2_iden_bn = BN1(128, 4)
        #####

        #####
        # Layer3 - self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer3_conv1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer3_bn1 = BN1(256, 4)
        self.layer3_conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.layer3_bn2 = BN1(256, 4)
        # self.layer3_iden_conv = torch.nn.Conv2d(128, 256, kernel_size=1, stride=2)
        # self.layer3_iden_bn = BN1(256, 4)
        #####

        #####
        # Layer4 - self.layer4 = self._make_layer(256, 512, stride=2)
        self.layer4_conv1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.layer4_bn1 = BN1(512, 4)
        self.layer4_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.layer4_bn2 = BN1(512, 4)
        # self.layer4_iden_conv = torch.nn.Conv2d(256, 512, kernel_size=1, stride=2)
        # self.layer4_iden_bn = BN1(512, 4)
        #####

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #####
        # Layer1
        # identity = x
        x = self.layer1_conv1(x)
        x = self.layer1_bn1(x)
        x = self.relu(x)
        
        x = self.layer1_conv2(x)
        x = self.layer1_bn2(x)
        # identity = self.layer1_iden_conv(identity)
        # identity = self.layer1_iden_bn(identity)
        # x += identity
        x = self.relu(x)

        # identity = x
        x = self.layer1_conv1(x)
        x = self.layer1_bn1(x)
        x = self.relu(x)
        x = self.layer1_conv2(x)
        x = self.layer1_bn2(x)
        # x += identity
        x = self.relu(x)
        #####

        #####
        # Layer2
        # identity = x
        x = self.layer2_conv1(x)
        x = self.layer2_bn1(x)
        x = self.relu(x)
        x = self.layer2_conv2(x)
        x = self.layer2_bn2(x)
        # identity = self.layer2_iden_conv(identity)
        # identity = self.layer2_iden_bn(identity)
        # x += identity
        x = self.relu(x)

        x = self.layer2_conv2(x)
        x = self.layer2_bn1(x)
        x = self.relu(x)
        x = self.layer2_conv2(x)
        x = self.layer2_bn2(x)
        # x += identity
        x = self.relu(x)
        #####

        #####
        # Layer3
        # identity = x
        x = self.layer3_conv1(x) 
        x = self.layer3_bn1(x)
        x = self.relu(x)
        x = self.layer3_conv2(x)
        x = self.layer3_bn2(x)
        # identity = self.layer3_iden_conv(identity)
        # identity = self.layer3_iden_bn(identity)
        # x += identity
        x = self.relu(x)

        x = self.layer3_conv2(x)
        x = self.layer3_bn1(x)
        x = self.relu(x)
        x = self.layer3_conv2(x)
        x = self.layer3_bn2(x)
        # x += identity
        x = self.relu(x)
        #####

        #####
        # Layer4
        # identity = x
        x = self.layer4_conv1(x)
        x = self.layer4_bn1(x)
        x = self.relu(x)
        x = self.layer4_conv2(x)
        x = self.layer4_bn2(x)
        # identity = self.layer4_iden_conv(identity)
        # identity = self.layer4_iden_bn(identity)
        # x += identity
        x = self.relu(x)

        x = self.layer4_conv2(x)
        x = self.layer4_bn1(x)
        x = self.relu(x)
        x = self.layer4_conv2(x)
        x = self.layer4_bn2(x)
        # x += identity
        x = self.relu(x)
        #####

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # same as x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

# image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
# img = load_and_preprocess_image(image_url)
# labels = load_labels()

model = ModResnet18()
module = torch_mlir.compile(model, torch.ones(1, 3, 224, 224), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)

# predictions(model.forward, jit_module.forward, img, labels)
# print("================ GIVEN MODEL =======================")
# resnet18 = models.resnet18(pretrained=True)
# # resnet18.train(False)
# # print(top3_possibilities(resnet18(img), labels))
# print(resnet18(img))
# print(model.forward(img))