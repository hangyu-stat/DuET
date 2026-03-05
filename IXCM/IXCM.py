import torch.nn as nn
import torchvision
import torch
from DWT import DWT_2D
import torch.nn.functional as F


class DenseNet121(nn.Module):
    
    def __init__(self, out_size=64, in_channels=1):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)

        original_conv = self.densenet121.features.conv0
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
            else:
                pass
        self.densenet121.features.conv0 = new_conv

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet18(nn.Module):
    
    def __init__(self, out_size=64, in_channels=1):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)

        original_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        self.resnet.conv1 = new_conv

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self, out_size=64, in_channels=1):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)

        original_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        self.resnet.conv1 = new_conv

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        return self.resnet(x)

class VGG16(nn.Module):

    def __init__(self, out_size=64, in_channels=1):
        super(VGG16, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=False)

        original_conv = self.vgg.features[0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=True
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                new_conv.bias = original_conv.bias
        
        self.vgg.features[0] = new_conv

        num_ftrs = 512 * 7 * 7 
        self.vgg.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        return self.vgg(x)

class MobileNetV2(nn.Module):
    def __init__(self, out_size=64, in_channels=1):
        super(MobileNetV2, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=False)

        original_conv = self.mobilenet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        self.mobilenet.features[0][0] = new_conv

        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        return self.mobilenet(x)

class ViT_B_16(nn.Module):
    def __init__(self, out_size=64, in_channels=1):
        super(ViT_B_16, self).__init__()
        self.vit = torchvision.models.vit_b_16(pretrained=False)

        original_conv = self.vit.conv_proj
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias
        
        self.vit.conv_proj = new_conv

        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit(x)

class NonNegativeELU(nn.Module):
    def forward(self, x):
        return F.elu(x) + 1.0 + 1e-6

class IXCNet(nn.Module):
    def __init__(self, backbone_name='densenet121', evi_vec_dim=64, wavename='haar', ydim=5, in_channels=1, img_dec=True):
        super(IXCNet, self).__init__()
        
        self.backbone_mapping = {
            'densenet121': DenseNet121,
            'resnet18': ResNet18,
            'resnet50': ResNet50,
            'vgg16': VGG16,
            'vit': ViT_B_16,
            'mobilenetv2': MobileNetV2
        }
        backbone_name = backbone_name.lower()
        if backbone_name not in self.backbone_mapping:
            raise ValueError(
                f"❌ Invalid backbone: '{backbone_name}'. "
                f"Supported backbones are: {list(self.backbone_mapping.keys())}"
            )
            
        print(f"✅ Building IXCNet with backbone: {backbone_name}")
        BackboneClass = self.backbone_mapping[backbone_name]
        
        self.evd = evi_vec_dim
        self.ydim = ydim
        self.wd_module = DWT_2D(wavename)
        self.imgv_dim_factor = 2
        self.dp_rate = 0.2
        self.img_dec = img_dec
        self.in_channels = in_channels
        self.shape_embed_layer = BackboneClass(out_size=self.imgv_dim_factor*self.evd, in_channels=self.in_channels)
        if self.img_dec:
            self.appearance_embed_layer = BackboneClass(out_size=self.imgv_dim_factor*self.evd, in_channels=self.in_channels)
        
        self.spev_embed_layer = nn.Sequential(
                nn.Linear(self.imgv_dim_factor * self.evd, self.imgv_dim_factor * self.evd), 
                nn.LayerNorm(self.imgv_dim_factor * self.evd),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate),
                nn.Linear(self.imgv_dim_factor * self.evd, self.evd * self.ydim)
            )
        self.snev_embed_layer = nn.Sequential(
                nn.Linear(self.imgv_dim_factor * self.evd, self.imgv_dim_factor * self.evd), 
                nn.LayerNorm(self.imgv_dim_factor * self.evd),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate),
                nn.Linear(self.imgv_dim_factor * self.evd, self.evd * self.ydim)
            )
        self.s_alpha_layers = nn.ModuleList()
        self.s_beta_layers = nn.ModuleList()
        if self.img_dec:
            self.apev_embed_layer = nn.Sequential(
                    nn.Linear(self.imgv_dim_factor * self.evd, self.imgv_dim_factor * self.evd), 
                    nn.LayerNorm(self.imgv_dim_factor * self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.imgv_dim_factor * self.evd, self.evd * self.ydim)
                )
            self.anev_embed_layer = nn.Sequential(
                    nn.Linear(self.imgv_dim_factor * self.evd, self.imgv_dim_factor * self.evd), 
                    nn.LayerNorm(self.imgv_dim_factor * self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.imgv_dim_factor * self.evd, self.evd * self.ydim)
                )
            self.a_alpha_layers = nn.ModuleList()
            self.a_beta_layers = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for i in range(self.ydim):
            self.s_alpha_layers.append(nn.Sequential(
                nn.Linear(self.evd * 2, self.evd * 2),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate / 2.0),
                nn.Linear(self.evd * 2, self.evd),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate / 4.0),
                nn.Linear(self.evd, self.evd // 2),
                nn.GELU(),
                nn.Linear(self.evd // 2, 1),
                NonNegativeELU()))
            self.s_beta_layers.append(nn.Sequential(
                nn.Linear(self.evd * 2, self.evd * 2),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate / 2.0),
                nn.Linear(self.evd * 2, self.evd),
                nn.GELU(),
                nn.Dropout(p=self.dp_rate / 4.0),
                nn.Linear(self.evd, self.evd // 2),
                nn.GELU(),
                nn.Linear(self.evd // 2, 1),
                NonNegativeELU()))

            if self.img_dec:
                self.a_alpha_layers.append(nn.Sequential(
                    nn.Linear(self.evd * 2, self.evd * 2),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate / 2.0),
                    nn.Linear(self.evd * 2, self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate / 4.0),
                    nn.Linear(self.evd, self.evd // 2),
                    nn.GELU(),
                    nn.Linear(self.evd // 2, 1),
                    NonNegativeELU()))
                self.a_beta_layers.append(nn.Sequential(
                    nn.Linear(self.evd * 2, self.evd * 2),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate / 2.0),
                    nn.Linear(self.evd * 2, self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate / 4.0),
                    nn.Linear(self.evd, self.evd // 2),
                    nn.GELU(),
                    nn.Linear(self.evd // 2, 1),
                    NonNegativeELU()))
                
                self.MLPs.append(nn.Sequential(
                    nn.Linear(4 * self.evd, 2 * self.evd),
                    nn.LayerNorm(2 * self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate),

                    nn.Linear(2 * self.evd, self.evd),
                    nn.LayerNorm(self.evd),
                    nn.GELU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),

                    nn.Linear(self.evd, self.evd // 2),
                    nn.LayerNorm(self.evd // 2),
                    nn.GELU(),
                    nn.Dropout(p=(self.dp_rate / 4.0)),

                    nn.Linear(self.evd // 2, 1),
                    nn.Sigmoid()
                ))
            else:
                self.MLPs.append(nn.Sequential(
                    nn.Linear(2 * self.evd, self.evd),
                    nn.LayerNorm(self.evd),
                    nn.GELU(),
                    nn.Dropout(p=self.dp_rate),

                    nn.Linear(self.evd, self.evd // 2),
                    nn.LayerNorm(self.evd // 2),
                    nn.GELU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),

                    nn.Linear(self.evd // 2, self.evd // 4),
                    nn.LayerNorm(self.evd // 4),
                    nn.GELU(),
                    nn.Dropout(p=(self.dp_rate / 4.0)),

                    nn.Linear(self.evd // 4, 1),
                    nn.Sigmoid()
                ))
    def forward(self, x):
        b = x.shape[0]
        
        if self.img_dec:
            s, lh, hl, hh = self.wd_module(x)
            a = lh + hl
        else:
            s = x
            
        s_embed = self.shape_embed_layer(s)
        spevs = self.spev_embed_layer(s_embed)
        snevs = self.snev_embed_layer(s_embed)
        
        if self.img_dec:
            a_embed = self.appearance_embed_layer(a)
            apevs = self.apev_embed_layer(a_embed)
            anevs = self.anev_embed_layer(a_embed)
        
        alphas = torch.empty((b, 0)).to(x.device)
        betas = torch.empty((b, 0)).to(x.device)
        outs = torch.empty((b, 0)).to(x.device)
        
        for i in range(self.ydim):
            MLP = self.MLPs[i]
            
            s_alpha_layer = self.s_alpha_layers[i]
            s_beta_layer = self.s_beta_layers[i]
            spev = spevs[:, i*self.evd:(i+1)*self.evd]
            snev = snevs[:, i*self.evd:(i+1)*self.evd]
            alpha_s = s_alpha_layer(torch.cat([spev, snev], dim=1)) + 1.0
            beta_s = s_beta_layer(torch.cat([spev, snev], dim=1)) + 1.0
            
            alphas = torch.cat([alphas, alpha_s], dim=1)
            betas = torch.cat([betas, beta_s], dim=1)
            
            if self.img_dec:
                a_alpha_layer = self.a_alpha_layers[i]
                a_beta_layer = self.a_beta_layers[i]
                apev = apevs[:, i*self.evd:(i+1)*self.evd]
                anev = anevs[:, i*self.evd:(i+1)*self.evd]
                alpha_a = a_alpha_layer(torch.cat([apev, anev], dim=1)) + 1.0
                beta_a = a_beta_layer(torch.cat([apev, anev], dim=1)) + 1.0
                
                y = MLP(torch.cat([spev, snev, apev, anev], dim=1))
                
                alphas = torch.cat([alphas, alpha_a], dim=1)
                betas = torch.cat([betas, beta_a], dim=1)
                
            else:
                y = MLP(torch.cat([spev, snev], dim=1))
            
            outs = torch.cat([outs, y], dim=1)

            
        return alphas, betas, outs



