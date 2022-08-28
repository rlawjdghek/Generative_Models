import torch
import torch.nn as nn
from torchvision import models

def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)
class LPIPS(nn.Module):
    def __init__(self, local_rank=0):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda(local_rank)
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda(local_rank)
    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load("metrics/lpips_weights.ckpt")
        else:
            state_dict = torch.load("metrics/lpips_weights.ckpt", map_location=torch.device("cpu"))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)
    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value
@torch.no_grad()
def calculate_lpips_given_images(group_of_images, local_rank=0):
    lpips = LPIPS(local_rank).eval().cuda(local_rank)
    lpips_values = []
    num_rand_outputs = len(group_of_images)  # [BS x 3 x H x W]가 들어있는 list
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_val = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_val.item()

if __name__=="__main__":
    import cv2
    import torch
    import torchvision.transforms as T
    from glob import glob
    paths = glob("/media/data1/jeonghokim/GANs/StarGANv2/20220624/eval_save_images/cat_to_dog/*")
    print(len(paths))
    imgs = []
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for path in paths:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = transform(img).cuda()
        imgs.append(img)
    print(calculate_lpips_given_images(imgs))

    
    

    