from itertools import chain
from math import e

import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection, pixel2world
from torchvision.transforms import Normalize
from .model_T_SD import Decoder, SlotAttention
from .model_general import dualRouteEncoder, SAMViT
from .utils import *
from segment_anything import sam_model_registry

class uorfNoGanTsamNewModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
        parser.add_argument('--shape_dim', type=int, default=32, help='Dimension of individual z latent per slot')
        parser.add_argument('--color_dim', type=int, default=8, help='Dimension of individual z latent per slot texture')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--supervision_size', type=int, default=64)
        parser.add_argument('--world_obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects in world coordinates')
        parser.add_argument('--obj_scale', type=float, default=3.5, help='Scale for locality on foreground objects in object-centric coordinates')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--weight_percept', type=float, default=0.006)
        parser.add_argument('--percept_in', type=int, default=100)
        parser.add_argument('--no_locality_epoch', type=int, default=1000)
        parser.add_argument('--locality_in', type=int, default=10)
        parser.add_argument('--locality_full', type=int, default=10)
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=64)
        parser.add_argument('--frustum_size_fine', type=int, default=128)
        parser.add_argument('--attn_decay_steps', type=int, default=2e5)
        parser.add_argument('--coarse_epoch', type=int, default=600)
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
        parser.add_argument('--fg_in_world', action='store_true', help='foreground objects are in world space')
        parser.add_argument('--dens_noise', type=float, default=1., help='Noise added to density may help in mitigating rank collapse')
        parser.add_argument('--invariant_in', type=int, default=0, help='when to start translation invariant decoding')
        parser.add_argument('--surface_loss', action='store_true', help='surface loss')
        parser.add_argument('--weight_surface', type=float, default=0.1)
        parser.add_argument('--surface_in', type=int, default=0)
        parser.add_argument('--load_pretrain', action='store_true', help='load partrained model')
        parser.add_argument('--load_pretrain_path', type=str, default=None)
        parser.add_argument('--only_decoder', action='store_true', help='')
        # parser.add_argument('--init_n_img_each_scene', type=int, default=3, help='number of images for each scene in the first epoch')
        # parser.add_argument('--init_n_epoch', type=int, default=0, help='number of epochs for the first epoch')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup',
                            sam_encoder=True)

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['recon', 'perc']
        if opt.surface_loss:
            self.loss_names.append('surface')
            self.surfaceLoss = surfaceLoss()
        self.set_visual_names()
        self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
        self.perceptual_net = get_perceptual_net().to(self.device)
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        
        z_dim = opt.shape_dim + opt.color_dim
        self.num_slots = opt.num_slots

        if not opt.preextract:
            sam_model = sam_model_registry[opt.sam_type](checkpoint=opt.sam_path)
            self.SAMViT = SAMViT(sam_model).cuda().eval()

        self.netEncoder = networks.init_net(dualRouteEncoder(input_nc=3, pos_emb=opt.pos_emb, bottom=opt.bottom, shape_dim=opt.shape_dim, color_dim=opt.color_dim),
                                                gpu_ids=self.gpu_ids, init_type='normal')

        self.netSlotAttention = networks.init_net(
            SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter, learnable_pos=not opt.no_learnable_pos), gpu_ids=self.gpu_ids, init_type='normal')
        
        self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=z_dim, n_layers=opt.n_layer,
                                                    locality_ratio=opt.world_obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality, 
                                                    project=opt.project, rel_pos=opt.relative_position, fg_in_world=opt.fg_in_world
                                                    ), gpu_ids=self.gpu_ids, init_type='xavier')

        if self.isTrain:
            if opt.load_pretrain: # load pretraine models, e.g., object NeRF decoder
                assert opt.load_pretrain_path is not None
                param_names1 = self.load_pretrain_networks(opt.load_pretrain_path, opt.epoch)
                # define two optimizers, one for keys in imcompatible.missing_keys, the other for the rest of the model
                param_names2 = [name for name, _ in self.netEncoder.named_parameters() if name not in param_names1] + \
                                [name for name, _ in self.netSlotAttention.named_parameters() if name not in param_names1] + \
                                [name for name, _ in self.netDecoder.named_parameters() if name not in param_names1]
                
                print('New params:', param_names1, '\n', 'Length:', len(param_names1))
                print('Loaded params:', param_names2, '\n', 'Length:', len(param_names2))
                
                # get corresponding parameters, may exist in either of the three models
                params1 = [v for k, v in self.netEncoder.named_parameters() if k in param_names1] + \
                            [v for k, v in self.netSlotAttention.named_parameters() if k in param_names1] + \
                            [v for k, v in self.netDecoder.named_parameters() if k in param_names1]
                params2 = [v for k, v in self.netEncoder.named_parameters() if k not in param_names1] + \
                            [v for k, v in self.netSlotAttention.named_parameters() if k not in param_names1] + \
                            [v for k, v in self.netDecoder.named_parameters() if k not in param_names1]
                print('Length:', len(params1), len(params2))
                self.optimizers = [optim.Adam(params1, lr=opt.lr), optim.Adam(params2, lr=opt.lr)]
                self.schedulers = [networks.get_scheduler(self.optimizers[0], opt), networks.get_freezeInit_scheduler(self.optimizers[1], opt)]

            else:
                requires_grad = lambda x: x.requires_grad
                params = chain(self.netEncoder.parameters(), self.netSlotAttention.parameters(), self.netDecoder.parameters())
                self.optimizer = optim.Adam(filter(requires_grad, params), lr=opt.lr)
                self.optimizers = [self.optimizer]
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        self.L2_loss = nn.MSELoss()

    def set_visual_names(self):
        n = self.opt.n_img_each_scene
        n_slot = self.opt.num_slots
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)] + \
                            ['slot{}_attn'.format(k) for k in range(n_slot)]

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.x = input['img_data'].to(self.device)
        self.x_large = input['img_data_large'].to(self.device)
        self.cam2world = input['cam2world'].to(self.device)
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        dens_noise = self.opt.dens_noise if (epoch <= self.opt.percept_in and self.opt.fixed_locality) else 0
        self.loss_recon = 0
        self.loss_perc = 0
        if self.opt.surface_loss:
            self.loss_surface = 0
        dev = self.x[0:1].device
        cam2world_viewer = self.cam2world[0]
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Encoding images
        with torch.no_grad():
            feature_map_sam = self.SAMViT(self.x_large[0:1].to(dev))  # BxC'xHxW, C': shape_dim (z_dim)
        # Encoder receives feature map from SAM and resized images as inputs
        feature_map = self.netEncoder(feature_map_sam,
            F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW, C: shape_dim+color_dim (z_dim+texture_dim)

        feat = feature_map.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC

        # Slot Attention
        z_slots, attn, fg_slot_position = self.netSlotAttention(feat)  # 1xKxC, 1xKxN (N=HxW), 1x(K-1)x2
        z_slots, attn, fg_slot_position = z_slots.squeeze(0), attn.squeeze(0), fg_slot_position.squeeze(0)  # KxC, KxN, K-1x2
        fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer)  # (K-1)x3
        
        K = attn.shape[0]

        cam2world = self.cam2world
        N = cam2world.shape[0]
        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            self.z_vals, self.ray_dir = z_vals, ray_dir
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([N, D, H, W, 3]), z_vals.view([N, H, W, D]), ray_dir.view([N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = self.x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir

        sampling_coor_fg = frus_nss_coor[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # Px3

        locality_ratio = 1 - min((epoch-self.opt.locality_in) / self.opt.locality_full, 1) * (1 - self.opt.obj_scale) if epoch >= self.opt.locality_in else None
        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        invariant = epoch >= self.opt.invariant_in
        # raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, fg_slot_nss_position, dens_noise=dens_noise, invariant=invariant)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
        raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, fg_slot_nss_position, dens_noise=dens_noise, invariant=invariant, locality_ratio=locality_ratio)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
        raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        masked_raws = masked_raws.view([K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
        rgb_map, _, weights = raw2outputs(raws, z_vals, ray_dir)
        # (NxHxW)x3, (NxHxW)
        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        self.loss_recon = self.L2_loss(x_recon, x)
        x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)
        if self.opt.surface_loss and epoch >= self.opt.surface_in:
            self.loss_surface = self.opt.weight_surface * self.surfaceLoss(weights)

        with torch.no_grad():
            attn = attn.detach().cpu()  # KxN
            H_, W_ = feature_map.shape[2], feature_map.shape[3]
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                setattr(self, 'x{}'.format(i), x[i])
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())
            setattr(self, 'attn', attn)

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])

                setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
        if self.opt.surface_loss:
            loss += self.loss_surface
        loss.backward()
        # self.loss_perc = self.loss_perc / self.weight_percept if self.weight_percept > 0 else self.loss_perc

    def optimize_parameters(self, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        self.forward(epoch)
        for opm in self.optimizers:
            opm.zero_grad()
        self.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
                if p.grad is not None and "bias" not in n:
                    with torch.no_grad():
                        layers.append(n)
                        avg_grads.append(p.grad.abs().mean().cpu().item())
        for opm in self.optimizers:
            opm.step()
        return layers, avg_grads

    def save_networks(self, surfix):
        """Save all the networks to the disk.
        Parameters:
            surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        super().save_networks(surfix)
        for i, opm in enumerate(self.optimizers):
            save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(opm.state_dict(), save_path)

        for i, sch in enumerate(self.schedulers):
            save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(sch.state_dict(), save_path)

    def load_networks(self, surfix):
        """Load all the networks from the disk.
        Parameters:
            surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
        """
        super().load_networks(surfix)

        if self.isTrain:
            for i, opm in enumerate(self.optimizers):
                load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the optimizer from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                opm.load_state_dict(state_dict)

            for i, sch in enumerate(self.schedulers):
                load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the lr scheduler from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                sch.load_state_dict(state_dict)


if __name__ == '__main__':
    pass