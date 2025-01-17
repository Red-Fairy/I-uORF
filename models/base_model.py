from json import load
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import time


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.exp_id)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'        

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_pretrain_networks(self, load_root, epoch):
        """
        load pretrained networks, some network may not exist,
        and keys in the pretrained networks may be fewer than current networks
        """
        unloaded_keys = []
        loaded_keys_frozen = []
        loaded_keys_trainable = []

        def load_module(module_name, load_method='unload'):
            # unload, load and freeze, load and train
            load_filename = '%s_net_%s.pth' % (epoch, module_name)
            load_path = os.path.join(load_root, load_filename)
            net = getattr(self, 'net' + module_name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if load_method == 'unload':
                for key, _ in net.named_parameters():
                    unloaded_keys.append(key)
            else:
                assert os.path.isfile(load_path), 'Pretrained network %s not exist' % load_path
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if 'fg_position' in state_dict and self.opt.one2four:
                    del state_dict['fg_position']
                # if 'fg_position' in state_dict:
                #     del state_dict['fg_position']
                if self.opt.no_load_sigma_mu or self.opt.diff_fg_init:
                    keys = list(state_dict.keys())
                    for key in keys:
                        if ('slots_logsigma' in key or 'slots_mu' in key) and 'bg' not in key:
                            del state_dict[key]
                if self.opt.learnable_slot_init:
                    keys = list(state_dict.keys())
                    for key in keys:
                        if 'slots_logsigma' in key or 'slots_mu' in key:
                            del state_dict[key]
                incompatible = net.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys and not self.opt.continue_train: # if continue train, ignore missing keys
                    for key in incompatible.missing_keys:
                        unloaded_keys.append(key)
                if incompatible.unexpected_keys and not self.opt.continue_train: # if continue train, ignore unexpected keys
                    assert False, 'Unexpected keys in pretrained network: %s' % incompatible.unexpected_keys
                    # add loaded keys to loaded_keys_frozen
                for key, _ in net.named_parameters():
                    if key not in incompatible.missing_keys or self.opt.continue_train:
                        if (load_method == 'load_train' or (self.opt.freeze_bg_only and 'f_' in key) or (self.opt.freeze_fg_only and 'b_' in key)):
                            loaded_keys_trainable.append(key)
                        else: #load_method == 'load_freeze':
                            loaded_keys_frozen.append(key)

        for name in self.model_names:
            load_type = getattr(self.opt, 'load_' + name.lower()) if hasattr(self.opt, 'load_' + name.lower()) else 'unload'
            load_module(name, load_type)
        
        return unloaded_keys, loaded_keys_frozen, loaded_keys_trainable
        # missing_names = ['SlotAttention']
        # if not self.opt.load_encoder and 'Encoder' in self.model_names:
        #     missing_names.append('Encoder')
        # if not self.opt.load_decoder and 'Decoder' in self.model_names:
        #     missing_names.append('Decoder')
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         load_filename = '%s_net_%s.pth' % (epoch, name)
        #         load_path = os.path.join(load_root, load_filename)
        #         net = getattr(self, 'net' + name)
        #         if isinstance(net, torch.nn.DataParallel):
        #             net = net.module
        #         if name in missing_names:
        #             for key, _ in net.named_parameters():
        #                 missing_keys.append(key)
        #             continue
        #         try:
        #             print('loading the model from %s' % load_path)
        #             if not os.path.isfile(load_path):
        #                 print(f'{load_path} not exist, skip')
        #                 continue
        #             state_dict = torch.load(load_path, map_location=str(self.device))
        #             incompatible = net.load_state_dict(state_dict, strict=False)
        #             if incompatible.missing_keys:
        #                 print(f'missing keys in state_dict: {incompatible.missing_keys}')
        #                 missing_keys.extend(incompatible.missing_keys)
        #             if incompatible.unexpected_keys:
        #                 print(f'Found unexpected keys in state_dict: {incompatible.unexpected_keys}, failed to load')
        #                 assert False
        #         except:
        #             print('Pretrained network %s not found. Keep training with initialized weights' % load_path)
        # return missing_keys

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                try:
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=self.device)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    if self.opt.isTrain:
                        result = net.load_state_dict(state_dict, strict=False)
                        print(result)
                    else:
                        net.load_state_dict(state_dict)
                    # print(result)
                    # try:
                    #     net.load_state_dict(state_dict, strict=not self.opt.not_strict)
                    # except:
                    #     del state_dict['fg_position']
                    #     net.load_state_dict(state_dict, strict=False)
                except FileNotFoundError:
                    assert False
                    print('not found: {} not found, skip {}'.format(load_path, name))
                except RuntimeError:
                    assert False
                    print('Size mismatch for {}, skip {}'.format(name, name))

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
