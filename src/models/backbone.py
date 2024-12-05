from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .cbam import CBAM_Module
from torch.hub import load_state_dict_from_url

class ResNetWithCBAM(ResNet):
    def __init__(self, block, layers):
        super(ResNetWithCBAM, self).__init__(block, layers)
        # Remove fully connected layer and avgpool
        self.fc = None
        self.avgpool = None

        # Add CBAM modules after each layer
        self.cbam1 = CBAM_Module(256)
        self.cbam2 = CBAM_Module(512)
        self.cbam3 = CBAM_Module(1024)
        self.cbam4 = CBAM_Module(2048)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def get_backbone_with_cbam():
    # Define the custom ResNet with CBAM
    resnet_cbam = ResNetWithCBAM(Bottleneck, [3, 4, 6, 3])

    # Load pretrained ResNet50 weights
    resnet_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')

    # Remove weights for fc and avgpool
    resnet_state_dict = {k: v for k, v in resnet_state_dict.items() if not k.startswith('fc.')}

    # Load weights into the custom ResNet
    resnet_cbam.load_state_dict(resnet_state_dict, strict=False)

    # Create the backbone with FPN
    backbone = BackboneWithFPN(
        resnet_cbam,
        return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256
    )

    return backbone