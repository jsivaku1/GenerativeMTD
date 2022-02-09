import argparse
import os
import torch

class TrainOptions:
# Parsing arguments
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self,parser):
        parser.add_argument("--dataset", "-f", nargs="?", default=os.path.join('Data', "imputed_SweatBinary.csv"), help="Dataset file used for training")
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument("--k", "-k", type=int, nargs="?", default=3, help="Number of nearest neighbors")
        parser.add_argument("--choose_best_k", action='store_true', help="Chooses best k automatically based on the dataset")
        parser.add_argument("--num_obs", "-n", type=int, nargs="?", default=100, help="Number of observations to generate")
        parser.add_argument("--epochs", "-e", type=int, nargs="?", default=100, help="Number of epochs to train")
        parser.add_argument("--batch_size", "-b", type=int, default=50, help="Batch size should be less than real data dim")
        parser.add_argument("--set_seed", "-s", type=int, nargs="?", default=1, help="Set random seed")
        parser.add_argument("--model", "-m", type=str, nargs="?", default='GenerativeMTD', help="GenerativeMTD | tablegan | veegan | ctgan | copulagan | TVAE ")
        # parser.add_argument("--cat_col", "-cc", type=str, nargs="?", default='', help="Categorical columns in the dataset")
        parser.add_argument("--target_col_ix", "-t", type=int, nargs="?", default='', help="Target column index")
        parser.add_argument("--ml_utility", "-u", type=str, nargs="?", default='classification', help="TSTR TRTS ML utility type classification | regression")
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        # # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt