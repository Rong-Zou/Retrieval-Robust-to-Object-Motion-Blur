import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils import seed_everything
import math


def resnet50_custom(
    pretrained=False,
    norm_type=None,
    image_size=[None, None],
    remove_maxpool_for_LN=False,
):
    """
    image_size is [H, W]
    norm_type is one of [None, 'NN', 'LN', 'IN', 'BN']
    None means no normalization, which is same as 'NN
    'LN' means layer normalization: torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
    'IN' means instance normalization: torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
    'BN' means batch normalization: torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

    """

    model = models.resnet50(pretrained=pretrained)

    i = 0
    feature_map_shapes_after_bn = []

    if norm_type == "LN":

        def forward_hook(module, input, output):
            feature_map_shapes_after_bn.append(output.shape[1:])

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.register_forward_hook(forward_hook)

        if remove_maxpool_for_LN:
            model.maxpool = nn.Identity()

        with torch.no_grad():
            model(torch.randn(2, 3, image_size[0], image_size[1]))

    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            if norm_type == "NN" or norm_type is None:
                setattr(model, name, nn.Identity())
            elif norm_type == "LN":
                shape = feature_map_shapes_after_bn[i]
                layer_norm = nn.LayerNorm(shape)
                setattr(model, name, layer_norm)
                i += 1
            elif norm_type == "IN":
                setattr(
                    model, name, nn.InstanceNorm2d(module.num_features)
                )  # , track_running_stats=True))
            elif norm_type == "BN_Original":
                pass
            elif norm_type == "BN_New":
                setattr(model, name, nn.BatchNorm2d(module.num_features))
            else:
                raise NotImplementedError

        if isinstance(module, nn.Sequential):
            for block in module:
                for sub_name, sub_module in block.named_children():
                    if isinstance(sub_module, nn.BatchNorm2d):
                        if norm_type == "NN" or norm_type is None:
                            setattr(block, sub_name, nn.Identity())
                        elif norm_type == "LN":
                            shape = feature_map_shapes_after_bn[i]
                            layer_norm = nn.LayerNorm(shape)
                            setattr(block, sub_name, layer_norm)
                            i += 1
                        elif norm_type == "IN":
                            # if you want to use running statistics for normalization during evaluation,
                            # you should set track_running_stats to True.
                            # If you always want to use batch statistics for normalization,
                            # you should set track_running_stats to False.
                            setattr(
                                block,
                                sub_name,
                                nn.InstanceNorm2d(sub_module.num_features),
                            )  # , track_running_stats=True))

                        elif norm_type == "BN_Original":
                            pass
                        elif norm_type == "BN_New":
                            setattr(
                                block, sub_name, nn.BatchNorm2d(sub_module.num_features)
                            )
                        else:
                            raise NotImplementedError
                    elif isinstance(sub_module, nn.Sequential):
                        for sub_sub_name, sub_sub_module in sub_module.named_children():
                            if isinstance(sub_sub_module, nn.BatchNorm2d):
                                if norm_type == "NN" or norm_type is None:
                                    setattr(sub_module, sub_sub_name, nn.Identity())
                                elif norm_type == "LN":
                                    shape = feature_map_shapes_after_bn[i]
                                    layer_norm = nn.LayerNorm(shape)
                                    setattr(sub_module, sub_sub_name, layer_norm)
                                    i += 1
                                elif norm_type == "IN":
                                    setattr(
                                        sub_module,
                                        sub_sub_name,
                                        nn.InstanceNorm2d(sub_sub_module.num_features),
                                    )  # , track_running_stats=True))
                                elif norm_type == "BN_Original":
                                    pass
                                elif norm_type == "BN_New":
                                    setattr(
                                        sub_module,
                                        sub_sub_name,
                                        nn.BatchNorm2d(sub_sub_module.num_features),
                                    )
                                else:
                                    raise NotImplementedError
    return model


class EncoderCNN(nn.Module):
    def __init__(
        self, version=2, pretrained=False, norm_type=None, image_size=[None, None]
    ):
        super(EncoderCNN, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.norm_type = norm_type
        self.image_size = image_size  # [H, W]
        assert self.norm_type in [None, "NN", "LN", "IN", "BN_Original", "BN_New"]
        # NN: no normalization
        # LN: layer normalization
        # IN: instance normalization
        # BN: batch normalization
        if version == 1:
            self.net = self.get_v1()
        elif version == 2:
            self.net = self.get_v2()
        elif version == 3:
            self.net = self.get_v3()
        else:
            self.net = self.get_v0()

        self.net_parts = list(self.net.children())

    def get_v0(self):  # original
        model = resnet50_custom(
            pretrained=self.pretrained,
            norm_type=self.norm_type,
            image_size=self.image_size,
        )
        return model  # N * 1000

    def get_v1(self):  # original without the last fc layer
        model = resnet50_custom(
            pretrained=self.pretrained,
            norm_type=self.norm_type,
            image_size=self.image_size,
        )
        modelc = nn.Sequential(*list(model.children())[:-2])
        return modelc  # N * 2048 * H/32 * W/32

    def get_v2(self):  # without maxpooling
        model = resnet50_custom(
            pretrained=self.pretrained,
            norm_type=self.norm_type,
            image_size=self.image_size,
            remove_maxpool_for_LN=True,
        )
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4])
        modelc3 = nn.Sequential(*list(model.children())[5])
        modelc4 = nn.Sequential(*list(model.children())[6])
        modelc5 = nn.Sequential(*list(model.children())[7])
        modelc = nn.Sequential(modelc1, modelc2, modelc3, modelc4, modelc5)
        return modelc  # N * 2048 * H/16 * W/16

    def get_v3(self):  # one more conv layer to reduce the dimension
        model = resnet50_custom(
            pretrained=self.pretrained,
            norm_type=self.norm_type,
            image_size=self.image_size,
        )
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        modelc3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        modelc = nn.Sequential(modelc1, modelc2, modelc3)
        return modelc  # N * 1024 * H/16 * W/16, 320/16 = 20, 240/16 = 15

    def forward(self, inputs, return_all_feature_maps=False):
        features = []

        if self.version != 2:
            features.append(self.net(inputs))
        else:
            if return_all_feature_maps:
                # pass inputs through each part of the network and get the output of each part and return all outputs
                for i in range(len(self.net_parts)):
                    inputs = self.net_parts[i](inputs)
                    features.append(inputs)

                # reverse the list and print the shape of each output
                features.reverse()

            else:
                # only return the output of the whole network
                features.append(self.net(inputs))

        return features


class pool_and_norm(nn.Module):
    def __init__(self, pooling="GeM", normalize=True, power=2, p=3, eps=1e-6):
        super(pool_and_norm, self).__init__()
        self.power = power

        if pooling == "GeM":
            self.pooling = GeM(p=p, eps=eps)
        elif pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError

        self.normalize = None
        if normalize:
            self.normalize = Normalize(power=self.power)

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        return x


class Normalize(nn.Module):
    def __init__(self, power=2, eps=1e-6):
        super(Normalize, self).__init__()
        self.power = power
        self.eps = eps

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + self.eps)
        a = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        assert torch.isclose(out, a).all()
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class Arcface(nn.Module):
    def __init__(self, in_feat, num_classes, s, m):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = s
        self._m = m

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer("t", torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return "in_features={}, num_classes={}, scale={}, margin={}".format(
            self.in_feat, self._num_classes, self._s, self._m
        )


if __name__ == "__main__":
    seed_everything()
    input = torch.randn(2, 3, 240, 320)

    blur_encoder1 = resnet50_custom(
        pretrained=True, norm_type="NN", image_size=(240, 320)
    )
    print(blur_encoder1)
    output = blur_encoder1(input)
    print(output.shape)

    blur_encoder2 = resnet50_custom(
        pretrained=True, norm_type="LN", image_size=(240, 320)
    )
    print(blur_encoder2)
    output = blur_encoder2(input)
    print(output.shape)

    blur_encoder3 = resnet50_custom(
        pretrained=True, norm_type="IN", image_size=(240, 320)
    )
    print(blur_encoder3)
    output = blur_encoder3(input)
    print(output.shape)

    blur_encoder4 = resnet50_custom(
        pretrained=True, norm_type="BN", image_size=(240, 320)
    )
    print(blur_encoder4)
    output = blur_encoder4(input)
    print(output.shape)
