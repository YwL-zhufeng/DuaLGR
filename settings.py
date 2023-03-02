import argparse

parser = argparse.ArgumentParser()

# acm
def acm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm', help='model_name')
    parser.add_argument('--train', type=bool, default=False, help='training mode')
    parser.add_argument('--dataset', type=str, default='acm', help='datasets: acm, dblp, texas, chameleon') # acm_hete_r{:.2f}_{}
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.8, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')

    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')

    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')

    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

# texas
def texas():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_texas', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.8, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=64, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=32, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args


# chameleon
def chameleon():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_chameleon', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=0.01, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.5, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.5, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=128, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=0, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

# dblp
def dblp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_dblp', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=5, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.8, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.5, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=500, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

def acm00():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm00', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.5, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

def acm01():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm01', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.5, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

def acm02():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm02', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.6, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

def acm03():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm03', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.6, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    args = parser.parse_args()
    return args

def acm04():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm04', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=10, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.5, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.5, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')

    args = parser.parse_args()
    return args

def acm05():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DuaLGR_acm05', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=10, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.8, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.3, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=1000, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=100, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')

    args = parser.parse_args()
    return args


def get_settings(dataset='acm'):
    args_dic = {
        'acm': acm(),
        'dblp': dblp(),
        'texas': texas(),
        'chameleon': chameleon(),
        'acm00': acm00(),
        'acm01': acm01(),
        'acm02': acm02(),
        'acm03': acm03(),
        'acm04': acm04(),
        'acm05': acm05(),
    }
    return args_dic[dataset]
