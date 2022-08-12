import os
import logging
import sys

def logging_setup(args):
    if not os.path.exists(args.log_dir):
        os.makedirs( args.log_dir )
    if not os.path.exists(args.model_dir):
        os.makedirs( args.model_dir )

    exp_name = args.dataset + '-' + args.model + '-e-'+str(args.epochs) + '-lr-'+str(args.lr)+ '-m-'+str(args.resnet_m)+\
               '-wd-'+ str(args.weight_decay) + '-b-'+str(args.batch_size)+'-K-' + str(args.K)+'-' +\
               '-dxy-' + str(int(args.constant_Dxy)) +\
               '-silu-' + str(int(args.use_silu)) +\
               '-res-' + str(int(args.use_res)) +\
               '-old-' + str(int(args.old_style)) +\
               '-fforg-' + str(int(args.use_f_for_g)) +\
               '-nof-' + str(int(args.no_f)) +\
               '-dt-' + str(args.dt) +\
               '-dx-' + str(args.dx) +\
               '-dy-' + str(args.dy) +\
               '-cDx-' + str(args.cDx) +\
               '-cDy-' + str(args.cDy) +\
               '-h0_h-' + str(int(args.init_h0_h))

    if args.pde_state != 0:
        exp_name += '-pde-' + str(args.pde_state) + '-'
    if args.custom_uv != '':
        exp_name += '-uv-' + str(args.custom_uv) + '-'
    if args.custom_dxy != '':
        exp_name += '-dxy-' + str(args.custom_dxy) + '-'

    if args.dataset in ['CIFAR-10', 'CIFAR-100']:
        exp_name += str(args.n1) + '-'+str(args.n2) + '-'+str(args.n3)+ '-'+str(args.n4)+'-sep-'+str(args.separable)
        if args.non_linear:
            exp_name += '-nonlin-'



    filename = str(f'./logs/{args.dataset}logfile-' + exp_name + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(filename=filename, encoding='utf-8'))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    return logging
