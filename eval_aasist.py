import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_eval
from data_utils_multiview import Dataset_var_eval, Dataset_var_eval2
from aasist_ssl import Model
from utils import reproducibility
from utils import read_metadata, read_metadata_eval, read_metadata_other
import numpy as np
from tqdm import tqdm
from dataio import pad
from torch import Tensor


class MyCollator(object):
    def __init__(self, **params):
        self.enable_chunking = params.get('enable_chunking', False)
        print("🐍 File: asvspoof5/train_multi_gpu.py | Line: 35 | __init__ ~ self.enable_chunking",
              self.enable_chunking)
        self.chunk_size = params.get('chunk_size', 64600)
        print("🐍 File: asvspoof5/train_multi_gpu.py | Line: 37 | __init__ ~ self.chunk_size", self.chunk_size)
        self.overlap_size = params.get(
            'overlap_size', 0)  # Default overlap size is 0
        print("🐍 File: asvspoof5/train_multi_gpu.py | Line: 39 | __init__ ~ self.overlap_size", self.overlap_size)

    def __call__(self, batch):
        if self.enable_chunking:
            return self.chunking(batch)
        return batch

    def chunking(self, batch):
        chunk_size = self.chunk_size
        overlap_size = self.overlap_size
        step_size = chunk_size - overlap_size

        split_data = []

        for x_inp, utt_id in batch:
            # Calculate number of chunks with overlap
            num_chunks = (len(x_inp) - overlap_size) // step_size

            # handle case where the utterance is smaller than overlap_size
            if num_chunks <= 0:
                padded_chunk = pad(
                    x=x_inp, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}_0"
                split_data.append((padded_chunk, chunk_id))
                continue

            for i in range(num_chunks):
                start = i * step_size
                end = start + chunk_size
                chunk = x_inp[start:end]
                chunk_id = f"{utt_id}_{i+1}"
                split_data.append((chunk, chunk_id))

            # Handle the case where the utterance is smaller than chunk_size
            if num_chunks * step_size + overlap_size < len(x_inp):
                start = num_chunks * step_size
                chunk = x_inp[start:]
                padded_chunk = pad(
                    x=chunk, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}_{num_chunks+1}"
                split_data.append((padded_chunk, chunk_id))

        # Convert to tensors (if they are not already tensors)
        x_inp_list, utt_id_list = zip(*split_data)

        x_inp_tensor = torch.stack(x_inp_list) if isinstance(
            x_inp_list[0], torch.Tensor) else torch.tensor(x_inp_list)
        return x_inp_tensor, utt_id_list


def produce_evaluation_file(dataset, model, device, save_path, batch_size, collator=None):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for i, (batch_x, utt_id) in enumerate(pbar):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]
                           ).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

    for f, cm in zip(fname_list, score_list):
        text_list.append('{} {}'.format(f, cm))
    del fname_list
    del score_list
    with open(save_path, 'a+') as fh:
        for i in range(0, len(text_list), 500):
            batch = text_list[i:i+500]
            fh.write('\n'.join(batch) + '\n')
    del text_list
    fh.close()
    print('Scores saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AASIST-W2V')
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/',
                        help='Change this to user\'s directory address of LA database')
    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/',
                        help='Change with path to user\'s LA database protocols directory address')
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
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch_size ')
    parser.add_argument('--cut', type=int, default=66800, metavar='N',
                        help='cut size ')
    parser.add_argument('--eval_chunking', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                        help='chunking eval')
    parser.add_argument('--chunk_size', type=int, default=16000, metavar='N',
                        help='chunk_size ')
    parser.add_argument('--overlap_size', type=int, default=8000, metavar='N',
                        help='overlap_size ')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Loading model
    model = Model(args, device)
    # model = nn.DataParallel(model)
    pretrained_dict = torch.load(args.ckpt_path, map_location=device)
    pretrained_dict = {k.replace('.module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {key.replace(
        "_orig_mod.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)

    # model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded : {}'.format(args.ckpt_path))

    eval_tracks = ['DF']
    model_tag = os.path.basename(args.ckpt_path).split('.')[0]
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)
    if args.dataset == 'asvspoof':
        for tracks in eval_tracks:
            if not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag)):
                prefix = 'ASVspoof_{}'.format(tracks)
                prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
                prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

                file_eval = read_metadata(dir_meta=os.path.join(
                    args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_eval=True)
                print('no. of eval trials', len(file_eval))
                if args.var:
                    print('var-length eval')
                    eval_set = Dataset_var_eval(list_IDs=file_eval, base_dir=os.path.join(
                        args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)), track=tracks)
                else:
                    eval_set = Dataset_eval(list_IDs=file_eval, base_dir=os.path.join(
                        args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)), track=tracks, cut=args.cut)
                produce_evaluation_file(
                    eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, model_tag), args.batch_size)
            else:
                print('Score file already exists')
    elif args.dataset == 'asvspoof5':
        print('ASVspoof 5 dataset')
        file_eval = read_metadata_eval(
            dir_meta=args.protocols_path, no_label=True)
        if args.var:
            print('var-length eval')
            eval_set = Dataset_var_eval2(
                list_IDs=file_eval, base_dir=args.database_path)
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
        else:
            eval_set = Dataset_eval(
                list_IDs=file_eval, base_dir=args.database_path, cut=args.cut, track='')
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
    else:
        print(f'other dataset: {args.dataset}')
        file_eval, _ = read_metadata_other(
            dir_meta=args.protocols_path, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_chunking = args.eval_chunking
        collator_params = {'enable_chunking': eval_chunking,
                           'chunk_size': args.chunk_size, 'overlap_size':  args.overlap_size}
        my_collator = MyCollator(
            **collator_params) if eval_chunking else None
        if args.var:
            print('var-length eval')
            eval_set = Dataset_var_eval2(
                list_IDs=file_eval, base_dir=args.database_path, format='')
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size, collator=my_collator)
        else:
            eval_set = Dataset_eval(
                list_IDs=file_eval, base_dir=args.database_path, cut=args.cut, track='', format='')
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size, collator=my_collator)
