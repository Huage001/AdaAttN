import torch
from PIL import Image
import numpy as np
import os
import random
import torch.nn as nn


def read_img(path, h, w):
    img = Image.open(path).convert('RGB').resize((w, h))
    img = (torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.).unsqueeze(0)
    return img


def save_img(img, path):
    img = (img[0].data.cpu().numpy().transpose((1, 2, 0)).clip(0, 1) * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class AttnAdaIN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AttnAdaIN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class AttnAdaINCos(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AttnAdaINCos, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        G_norm = torch.sqrt((G ** 2).sum(1).view(b, 1, -1))
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h)
        F_norm = torch.sqrt((F ** 2).sum(1).view(b, -1, 1))
        F = F.permute(0, 2, 1)
        S = torch.relu(torch.bmm(F, G) / (F_norm + 1e-5) / (G_norm + 1e-5) + 1)
        # S: b, n_c, n_s
        S = S / (S.sum(dim=-1, keepdim=True) + 1e-5)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs):
        return self.decoder(cs)


def get_key(feats):
    results = []
    _, _, h, w = feats[-1].shape
    for i in range(len(feats) - 1):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[-1]))
    return torch.cat(results, dim=1)


def main():
    src_root = 'video_root'
    transformer_path = 'attn_adain_video/latest_net_transformer.pth'
    decoder_path = 'attn_adain_video/latest_net_decoder.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    image_encoder.load_state_dict(torch.load('vgg_normalised.pth'))
    enc_layers = list(image_encoder.children())
    enc_1 = nn.Sequential(*enc_layers[:4]).to(device)
    enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)
    enc_3 = nn.Sequential(*enc_layers[11:18]).to(device)
    image_encoder_layers = [enc_1, enc_2, enc_3]
    for layer in image_encoder_layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False
    transformer = AttnAdaINCos(in_planes=256, key_planes=256 + 128 + 64, max_sample=256 * 256).to(device)
    decoder = Decoder().to(device)
    transformer.load_state_dict(torch.load(transformer_path))
    decoder.load_state_dict(torch.load(decoder_path))
    transformer.eval()
    decoder.eval()
    for param in transformer.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    def encode_with_intermediate(img):
        results = [img]
        for i in range(len(image_encoder_layers)):
            func = image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    style_root = 'style'
    output_root = 'result_video'
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    for style_name in os.listdir(style_root):
        tgt_root = os.path.join(output_root, style_name[:style_name.rfind('.')])
        style_path = os.path.join(style_root, style_name)
        if not os.path.exists(tgt_root):
            os.mkdir(tgt_root)
        style = read_img(style_path, 512, 512).to(device)
        style_feats = encode_with_intermediate(style)
        seed = random.randint(0, 1000000)
        with torch.no_grad():
            for folder in sorted(os.listdir(src_root)):
                print('Processing Video %s...' % folder)
                src_folder_path = os.path.join(src_root, folder)
                if os.path.isdir(src_folder_path):
                    tgt_folder_path = os.path.join(tgt_root, folder)
                    if not os.path.exists(tgt_folder_path):
                        os.mkdir(tgt_folder_path)
                    for idx, name in enumerate(sorted(os.listdir(src_folder_path))):
                        frame_path = os.path.join(src_folder_path, name)
                        frame = read_img(frame_path, 256, 512).to(device)
                        frame_feats = encode_with_intermediate(frame)
                        result = decoder(transformer(frame_feats[-1], style_feats[-1],
                                                     get_key(frame_feats), get_key(style_feats), seed))
                        save_img(result, os.path.join(tgt_folder_path, name))
                        if idx % 5 == 0:
                            print('\tFrame %d finished!' % idx)


if __name__ == '__main__':
    main()

