import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks


class AdaAttNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', required=True, help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3', action='store_true',
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer', action='store_true',
                            help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=0., help='weight for L2 content loss')
            parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_local', type=float, default=3.,
                                help='weight for attention weighted style loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
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
        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'transformer']
        parameters = []
        self.max_sample = 64 * 64
        if opt.skip_connection_3:
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                              max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer:
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512
        transformer = networks.Transformer(
            in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
        decoder = networks.Decoder(opt.skip_connection_3)
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())
        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        if self.isTrain:
            self.loss_names = ['content', 'global', 'local']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_local = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):
        self.c_feats = self.encode_with_intermediate(self.c)
        self.s_feats = self.encode_with_intermediate(self.s)
        if self.opt.skip_connection_3:
            c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], self.get_key(self.c_feats, 2, self.opt.shallow_layer),
                                                   self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
        else:
            c_adain_feat_3 = None
        cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
                                  self.get_key(self.c_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.c_feats, 4, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
        self.cs = self.net_decoder(cs, c_adain_feat_3)

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_local = torch.tensor(0., device=self.device)
        if self.opt.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i], std * networks.mean_variance_norm(self.c_feats[i]) + mean)

    def compute_losses(self):
        stylized_feats = self.encode_with_intermediate(self.cs)
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global
        
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_global + self.loss_local
        loss.backward()
        self.optimizer_g.step()

