import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_eval
from data_utils_multiview import Dataset_var_eval, Dataset_var_eval2
from model import Model
from utils import reproducibility
from utils import read_metadata, read_metadata_eval
import numpy as np
from tqdm import tqdm

def produce_emb_file(dataset, model, device, save_path, batch_size=10, is_embedding=False, last_emb=False):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

    model.eval()

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            batch_x = batch_x.to(device)
            batch_emb = model(batch_x, is_embedding=is_embedding, last_hidden_state=last_emb)
            # print(batch_emb.shape)
            # import sys
            # sys.exit(1)
            fname_list.extend(utt_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Then each emb should be save in a file with name is utt_id
            for f, emb in zip(fname_list,batch_emb):
                # normalize filename
                f = f.split('/')[-1].split('.')[0] # utt id only
                save_path_utt = os.path.join(save_path, f)
                np.save(save_path_utt, emb.data.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s directory address of LA database')
    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--ckpt_path', type=str, 
                    help='path to the model weigth')
    parser.add_argument('--dataset', type=str, 
                    help='path to the model weigth', default='asvspoof')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    parser.add_argument('--var', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                        help='var-length')
    parser.add_argument('--exact_emb', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                        help='exact_emb')
    parser.add_argument('--last_emb', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                        help='last_emb')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch_size ')
    parser.add_argument('--cut', type=int, default=66800, metavar='N',
                    help='cut size ')
    parser.add_argument('--save_path', type=str, default='emb/', help='Change this to user\'s directory address of LA database')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to extract')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # Loading model
    model = Model(args,device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded : {}'.format(args.ckpt_path))
    
    # Join save path with cut
    args.save_path = os.path.join(args.save_path, str(args.cut) if args.cut > 0 else 'full') 
    # Create save path if not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    eval_tracks=['DF']
    tracks = 'DF'
    prefix      = 'ASVspoof_{}'.format(tracks)
    prefix_2021 = 'ASVspoof2021.{}'.format(tracks)
    if args.num_samples != -1:
        file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format( prefix,prefix_2021)), is_eval=True)[:args.num_samples]
    else:
        file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format( prefix,prefix_2021)), is_eval=True)[:args.num_samples]
    print('no. of eval trials',len(file_eval))
    eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)),track=tracks, cut=args.cut)
    produce_emb_file(eval_set, model, device, args.save_path, args.batch_size, is_embedding=args.exact_emb, last_emb=args.last_emb)
    print('Embedding saved to {}'.format(args.save_path))