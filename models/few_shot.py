import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist, cosine_distance


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# 模型
# 计算episode、acc和loss
class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    """
    sample的格式为：
    {
    	"class":长度为10的list,
    	"xs":torch.Size([5, 5, 1, 28, 28]),    # 5-way 5-shot 1*28*28像素
    	"xq":torch.Size([5, 15, 1, 28, 28])     # 5-way 15-query
    }
    
    """

    def type_eva(self, sample):
        z = self.encoder.forward(sample)  # z就是一个[batch_size,1,28,28]的tensor
        return z

    def loss(self, sample):
        xs = Variable(sample['xs'])  # support torch.Size([5, 5, 1, 28, 28])
        xq = Variable(sample['xq'])  # query torch.Size([5, 15, 1, 28, 28])

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)  # 60,5,5
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])],
                      0)
        z = self.encoder.forward(x)  # z.size([600, 2048])
        z_dim = z.size(-1)  # 展平,z_dim = 2048
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim)  # 不取平均，保留每张图片的特征

        zq = z[n_class * n_support:]  # 取出上面拼接的特征中属于query的部分，即300张query的特征，300,2048
        dists = euclidean_dist(zq, z_proto)  

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # print('log_p_y:', log_p_y.size())
        # 计算损失，先在第2+1个维度上寻找对应标签的距离，例如类1的样本2标签是5，取出它距离原型1的距离，这就是这个样本的产生的loss，然后对所有样本求平均loss
        # print('loss_val:', loss_val)
        # 计算预测query的标签。根据距离结果，选最小的距离，但是之前-dist，所以这里就是max。
        _, y_hat = log_p_y.max(2)  # y_hat[60,5],60-way 5-query,即预测的每张query的标签
        # y_hat = torch.floor(y_hat / n_support)
        # print('y_hat:', y_hat, 'dists:', dists)
        # .max(2)表示在第2维度进行比较，max返回值有两个，一个是最大值，另一个是坐标。y_hat取的就是坐标
        # print('y_hat:', y_hat.size(), y_hat)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()  # y_hat与target_inds相等的元素所占的比例

        # log_p_y_less = torch.Tensor(n_class, n_query, n_class).zero_()
        # log_p_y_less = log_p_y_less.cuda()
        # for i in range(n_class):
        #     for j in range(n_query):
        #         for k in range(n_class):
        #             s = log_p_y[i, j, 10 * k]
        #             for g in range(n_support):
        #                 if s >= log_p_y[i, j, 10 * k + g]:
        #                     s = log_p_y[i, j, 10 * k + g]
        #             log_p_y_less[i, j, k] = s  # 第i类，第j张query，与第k类的所有support中的最小距离
        # # print(log_p_y_less.size())
        # loss_val = -log_p_y_less.gather(2, target_inds).squeeze().view(-1).mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


# class Residual(nn.Module):  # Resnet18中用的BasicBlock
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(Residual, self).__init__()
#         self.stride = stride
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         if in_channels != out_channels:  # 如果输入的通道和输出的通道数不一样，则使用1×1的卷积残差块，也就是shortcut
#             self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#             self.bn = nn.BatchNorm2d(out_channels)
#         else:
#             self.conv1x1 = None
#
#     def forward(self, x):
#         o1 = self.relu(self.bn1(self.conv1(x)))
#         o2 = self.bn2(self.conv2(o1))
#         if self.conv1x1:
#             x = self.bn(self.conv1x1(x))
#         out = self.relu(o2 + x)
#         return out


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    # x_dim = kwargs['x_dim']  # [1,28,28] --> [3,224,224]
    # hid_dim = kwargs['hid_dim']  # 64
    # z_dim = kwargs['z_dim']  # 64
    #
    # def conv_block(in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )


    encoder = nn.Sequential(
        # #resnet18
        # nn.Sequential(
        #     nn.Conv2d(x_dim[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ),  # Resnet layer0
        # nn.Sequential(
        #     Residual(64, 64),
        #     Residual(64, 64)
        # ),
        # nn.Sequential(
        #     Residual(64, 128, stride=2),
        #     Residual(128, 128)
        # ),
        # nn.Sequential(
        #     Residual(128, 256, stride=2),
        #     Residual(256, 256)
        # ),
        # nn.Sequential(
        #     Residual(256, 512, stride=2),
        #     Residual(512, 512)
        # ),
        # nn.AdaptiveAvgPool2d(output_size=(1, 1)),

        # conv_block(x_dim[0], hid_dim),  # (1,64)
        # conv_block(hid_dim, hid_dim),  # (64,64)
        # conv_block(hid_dim, hid_dim),  # (64,64)
        # conv_block(hid_dim, z_dim),  # (64,64)
        # Flatten()  # (64,1,7,7)


        nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 1为通道数，RGB就改为3
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        Flatten()


        # Conv3x3(3, 256, 5, 2, 2, False, pad_mode='reflect'),
        # Conv3x3(256, 256, 3, 1, 0, False),
        # ConvResBlock(256 * 1, 256 * 2, 4, 2, 0, 10, False),
        # ConvResBlock(256 * 2, 256 * 4, 4, 2, 0, 10, False),
        # ConvResBlock(256 * 4, 256 * 8, 2, 2, 0, 10, False),
        # MaybeBatchNorm2d(256 * 8, True, False),
        # ConvResBlock(256 * 8, 256 * 8, 3, 1, 0, 10, False),
        # ConvResBlock(256 * 8, 256 * 8, 3, 1, 0, 10, False),
        # ConvResNxN(256 * 8, 2048, 3, 1, 0, False),
        # MaybeBatchNorm2d(2048, True, True)


    )
    
    return Protonet(encoder)
