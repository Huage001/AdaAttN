#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import cv2


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


def get_key(feats, last_layer_idx):
    results = []
    _, _, h, w = feats[last_layer_idx].shape
    for i in range(last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[last_layer_idx]))
    return torch.cat(results, dim=1)


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

    def forward(self, content, style, content_key, style_key, seed=None, content_mask=None, style_mask=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if style_mask is not None:
            style_mask = nn.functional.interpolate(
                style_mask, size=(h_g, w_g), mode='nearest').view(b, 1, w_g * h_g).contiguous()
        else:
            style_mask = torch.ones(b, 1, w_g * h_g, device=style.device)
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_mask = style_mask[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        if content_mask is not None:
            content_mask = nn.functional.interpolate(
                content_mask, size=(h, w), mode='nearest').view(b, 1, w * h).permute(0, 2, 1).contiguous()
        else:
            content_mask = torch.ones(b, w * h, 1, device=content.device)
        S = torch.bmm(F, G)
        style_mask = 1. - style_mask
        attn_mask = torch.bmm(content_mask, style_mask)
        S = S.masked_fill(attn_mask.bool(), -1e15)
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


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AttnAdaIN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AttnAdaIN(in_planes=in_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key,
                content5_1_key, style5_1_key, seed=None, content_mask=None, style_mask=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(
                content4_1, style4_1, content4_1_key, style4_1_key, seed, content_mask, style_mask) +
            self.upsample5_1(self.attn_adain_5_1(
                content5_1, style5_1, content5_1_key, style5_1_key, seed, content_mask, style_mask))))


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256, 256, (3, 3)),
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

    def forward(self, cs, c_adain_3_feat):
        cs = self.decoder_layer_1(cs)
        cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs


# In[2]:
transformer_path = 'models/attn_adain_without_ss_idt/latest_net_transformer.pth'
decoder_path = 'models/attn_adain_without_ss_idt/latest_net_decoder.pth'
attn_adain_3_path = 'models/attn_adain_without_ss_idt/latest_net_attn_adain_3.pth'
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
image_encoder.load_state_dict(torch.load('../vgg_normalised.pth'))
enc_layers = list(image_encoder.children())
enc_1 = nn.Sequential(*enc_layers[:4]).to(device)
enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)
enc_3 = nn.Sequential(*enc_layers[11:18]).to(device)
enc_4 = nn.Sequential(*enc_layers[18:31]).to(device)
enc_5 = nn.Sequential(*enc_layers[31:44]).to(device)
image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
for layer in image_encoder_layers:
    layer.eval()
    for param in layer.parameters():
        param.requires_grad = False
transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(device)
decoder = Decoder().to(device)
attn_adain_3 = AttnAdaIN(in_planes=256, key_planes=256 + 128 + 64, max_sample=256 * 256).to(device)
transformer.load_state_dict(torch.load(transformer_path))
decoder.load_state_dict(torch.load(decoder_path))
attn_adain_3.load_state_dict(torch.load(attn_adain_3_path))
transformer.eval()
decoder.eval()
attn_adain_3.eval()
for param in transformer.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False
for param in attn_adain_3.parameters():
    param.requires_grad = False


def encode_with_intermediate(img):
    results = [img]
    for i in range(len(image_encoder_layers)):
        func = image_encoder_layers[i]
        results.append(func(results[-1]))
    return results[1:]


# In[3]:

content_path = '../content_more/img50.jpg'
style_path = '../style/wreck.jpg'
content_name = os.path.basename(content_path)
style_name = os.path.basename(style_path)
tgt_path = content_name[:content_name.rfind('.')] + '_' + style_name[:style_name.rfind('.')] + '.jpg'
# In[4]:

btn_down = False
point_a = (0, 0)
x_max = 0
x_min = 1e10
y_max = 0
y_min = 1e10
content_im = cv2.imread(content_path)
content_im = cv2.resize(content_im, (512, 512))
content_result = content_im.copy()
h, w = content_result.shape[:2]
content_canvas = np.zeros([h, w, 3], np.uint8)
content_mask = np.zeros([h, w], np.uint8)


def mouse_content_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        global content_mask, content_result
        content_result = content_im.copy()
        h, w = content_result.shape[:2]
        content_mask = np.zeros([h + 2, w + 2], np.uint8)
        cv2.floodFill(content_result, content_mask, (x, y), (255, 255, 255),
                      (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
        content_mask = content_mask[1:-1, 1:-1]
        cv2.imshow('Content Image', content_result)


def mouse_content_paint(event, x, y, flags, param):
    global btn_down, point_a, x_max, x_min, y_max, y_min
    if event == cv2.EVENT_LBUTTONUP and btn_down:
        btn_down = False
        cv2.line(content_result, point_a, (x, y), (255, 255, 255))
        cv2.line(content_canvas, point_a, (x, y), (255, 255, 255))
        point_a = (x, y)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        cv2.imshow('Content Image', content_result)
    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        cv2.line(content_result, point_a, (x, y), (255, 255, 255))
        cv2.line(content_canvas, point_a, (x, y), (255, 255, 255))
        point_a = (x, y)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        cv2.imshow('Content Image', content_result)
    elif event == cv2.EVENT_LBUTTONDOWN:
        btn_down = True
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        point_a = (x, y)


print('Please choose an interactive mode for content image:')
print('\t1: Click Mode')
print('\t2: Paint Mode')
print('\tOther: No Interaction')
option = input()
if option == '1':
    cv2.namedWindow('Content Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Content Image', content_im)
    cv2.setMouseCallback('Content Image', mouse_content_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(content_name[:content_name.rfind('.')] + '_interactive.jpg', content_result)
elif option == '2':
    cv2.namedWindow('Content Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Content Image', content_im)
    cv2.setMouseCallback('Content Image', mouse_content_paint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    content_mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(content_canvas, content_mask, (int((x_max + x_min) / 2), int((y_max + y_min) / 2)), (255, 255, 255),
                  (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    content_mask = content_mask[1:-1, 1:-1]
    cv2.imwrite(content_name[:content_name.rfind('.')] + '_interactive.jpg', content_result)

# In[5]:
btn_down = False
point_a = (0, 0)
x_max = 0
x_min = 1e10
y_max = 0
y_min = 1e10
style_im = cv2.imread(style_path)
style_im = cv2.resize(style_im, (512, 512))
style_result = style_im.copy()
h, w = style_result.shape[:2]
style_canvas = np.zeros([h, w, 3], np.uint8)
style_mask = np.zeros([h, w], np.uint8)


def mouse_style_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        global style_mask, style_result
        style_result = style_im.copy()
        h, w = style_result.shape[:2]
        style_mask = np.zeros([h + 2, w + 2], np.uint8)
        cv2.floodFill(style_result, style_mask, (x, y), (255, 255, 255),
                      (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
        style_mask = style_mask[1:-1, 1:-1]
        cv2.imshow('Style Image', style_result)


def mouse_style_paint(event, x, y, flags, param):
    global btn_down, point_a, x_max, x_min, y_max, y_min
    if event == cv2.EVENT_LBUTTONUP and btn_down:
        btn_down = False
        cv2.line(style_result, point_a, (x, y), (255, 255, 255))
        cv2.line(style_canvas, point_a, (x, y), (255, 255, 255))
        point_a = (x, y)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        cv2.imshow('Style Image', style_result)
    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        cv2.line(style_result, point_a, (x, y), (255, 255, 255))
        cv2.line(style_canvas, point_a, (x, y), (255, 255, 255))
        point_a = (x, y)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        cv2.imshow('Style Image', style_result)
    elif event == cv2.EVENT_LBUTTONDOWN:
        btn_down = True
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        point_a = (x, y)


print('Please choose an interactive mode for style image:')
print('\t1: Click Mode')
print('\t2: Paint Mode')
print('\tOther: No Interaction')
option = input()
if option == '1':
    cv2.namedWindow('Style Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Style Image', style_im)
    cv2.setMouseCallback('Style Image', mouse_style_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(style_name[:style_name.rfind('.')] + '_interactive.jpg', style_result)
elif option == '2':
    cv2.namedWindow('Style Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Style Image', style_im)
    cv2.setMouseCallback('Style Image', mouse_style_paint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    style_mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(style_canvas, style_mask, (int((x_max + x_min) / 2), int((y_max + y_min) / 2)), (255, 255, 255),
                  (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    style_mask = style_mask[1:-1, 1:-1]
    cv2.imwrite(style_name[:style_name.rfind('.')] + '_interactive.jpg', style_result)

# In[6]:

with torch.no_grad():
    style = read_img(style_path, h, w).to(device)
    content = read_img(content_path, h, w).to(device)
    style_mask = torch.from_numpy(style_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    content_mask = torch.from_numpy(content_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    c_feats = encode_with_intermediate(content)
    s_feats = encode_with_intermediate(style)
    c_adain_feat_3 = attn_adain_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2), None,
                                  content_mask, style_mask)
    cs = transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                     get_key(c_feats, 4), get_key(s_feats, 4), None, content_mask, style_mask)
    cs = decoder(cs, c_adain_feat_3)
    save_img(cs, tgt_path)

