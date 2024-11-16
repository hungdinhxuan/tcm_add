import argparse
import sys
import os
from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_eval
from data_utils_multiview import Dataset_train as Dataset_train_multiview
from utils import read_metadata, read_metadata_eval, read_metadata_other
from data_utils_multiview import Dataset_var_eval, Dataset_var_eval2
from model import Model
from utils import reproducibility
from utils import read_metadata
import numpy as np
from tqdm import tqdm
from collate_fn import multi_view_collate_fn

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        target = torch.LongTensor(batch_y).to(device)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, _ = model(batch_x)
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        i=i+1
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('\n{} - {} - {} '.format(epoch, str(test_accuracy)+'%', val_loss))
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []

    with torch.no_grad():
        for batch_x,utt_id in data_loader:
            batch_x = batch_x.to(device)
            batch_out, _ = model(batch_x)
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

def train_epoch(train_loader, model, lr,optim, device):
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        batch_x, batch_y = batch

        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, _ = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        pbar.set_description(f"Epoch {epoch}: cls_loss {batch_loss.item()}")
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        i=i+1
    sys.stdout.flush()

def train_epoch_multiview(
    train_loader, 
    model, 
    optimizer, 
    device, 
    weighted_views: Dict[str, float] = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
):
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Initialize metrics for each view
    running_losses = {f'view_{v}': 0.0 for v in weighted_views.keys()}
    running_corrects = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_samples = {f'view_{v}': 0 for v in weighted_views.keys()}
    
    pbar = tqdm(train_loader)
    for batch in pbar:
        total_loss = 0.0
        batch_metrics = {}
        
        # Process each view
        for view_idx, (inputs, labels) in batch.items():
            view = str(view_idx)  # Convert view index to string
            
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.view(-1).type(torch.int64).to(device)
            batch_size = inputs.size(0)
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels) * weighted_views[view]
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()
            
            # Update running metrics
            running_losses[f'view_{view}'] += loss.item() * batch_size
            running_corrects[f'view_{view}'] += correct
            running_samples[f'view_{view}'] += batch_size
            
            # Accumulate total loss
            total_loss += loss
            
            # Calculate batch metrics for progress bar
            batch_metrics[f'loss_view_{view}'] = loss.item()
            batch_metrics[f'acc_view_{view}'] = correct / batch_size
        
        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        
        # Update progress bar
        desc = f"Loss: {total_loss.item():.4f} | " + " | ".join(
            f"V{k[-1]} Loss: {v:.4f}" for k, v in batch_metrics.items() if 'loss' in k
        ) + " | " + " | ".join(
            f"V{k[-1]} Acc: {v:.4f}" for k, v in batch_metrics.items() if 'acc' in k
        )
        pbar.set_description(desc)
    
    # Calculate epoch metrics
    epoch_metrics = {}
    for view in weighted_views.keys():
        view_key = f'view_{view}'
        if running_samples[view_key] > 0:
            epoch_metrics[f'train_loss/{view_key}'] = running_losses[view_key] / running_samples[view_key]
            epoch_metrics[f'train_acc/{view_key}'] = running_corrects[view_key] / running_samples[view_key]
    
    return epoch_metrics

def dev_epoch_multiview(dev_loader, model, device, weighted_views: Dict[str, float] = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}):
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    num_total = 0
    metrics = {
        view: {'correct': 0, 'total': 0, 'loss': 0.0} 
        for view in weighted_views.keys()
    }
    
    # Set objective (Loss) function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='sum')
    
    with torch.no_grad():
        pbar = tqdm(dev_loader)
        for batch in pbar:
            loss_detail = {}
            batch_total_loss = 0.0
            
            for view, (batch_x, batch_y) in batch.items():
                view = str(view)  # Convert view to string for indexing
                
                # Move data to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                batch_out, _ = model(batch_x)
                
                # Calculate loss
                batch_loss = criterion(batch_out, batch_y) * weighted_views[view]
                
                # Calculate predictions
                pred = batch_out.max(1)[1]
                
                # Update metrics for this view
                batch_size = batch_x.size(0)
                correct = pred.eq(batch_y).sum().item()
                
                metrics[view]['correct'] += correct
                metrics[view]['total'] += batch_size
                metrics[view]['loss'] += batch_loss.item()
                
                # Store batch metrics
                loss_detail[f'dev_loss/view_{view}'] = batch_loss.item() / batch_size
                loss_detail[f'dev_acc/view_{view}'] = correct / batch_size
                
                batch_total_loss += batch_loss.item()
                num_total += batch_size
            
            # Update progress bar with current batch metrics
            pbar.set_description(f"Loss: {batch_total_loss/batch_size:.4f}")
    
    # Calculate final metrics for each view
    results = {}
    total_loss = 0.0
    total_accuracy = 0.0
    
    for view in metrics:
        view_loss = metrics[view]['loss'] / metrics[view]['total']
        view_accuracy = 100. * metrics[view]['correct'] / metrics[view]['total']
        
        results[f'dev_loss/view_{view}'] = view_loss
        results[f'dev_acc/view_{view}'] = view_accuracy
        
        total_loss += view_loss * weighted_views[view]
        total_accuracy += view_accuracy * weighted_views[view]
    
    # Calculate weighted averages
    total_weight = sum(weighted_views.values())
    avg_loss = total_loss / total_weight
    avg_accuracy = total_accuracy / total_weight
    
    print(f'\nAverage Loss: {avg_loss:.4f}')
    print(f'Average Accuracy: {avg_accuracy:.2f}%')
    
    for view in metrics:
        print(f'View {view} - Accuracy: {results[f"dev_acc/view_{view}"]:.2f}% '
              f'Loss: {results[f"dev_loss/view_{view}"]:.4f}')
    
    return avg_loss, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASVspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--dataset', type=str, default='asvspoof',
                        help='Comment to describe the saved model')
    
    parser.add_argument('--is_multiview', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Multiview train')
    
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'Conformer_w_TCM_{}_{}_{}_ES{}_H{}_NE{}_KS{}_AUG{}_w_sin_pos'.format(
        track, args.loss, args.lr,args.emb_size, args.heads, args.num_encoders, args.kernel_size, args.algo)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    print('Model tag: '+ model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
     
    if args.dataset == 'asvspoof':
        # define train dataloader
        label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)), is_eval=False)
        print('no. of training trials',len(files_id_train))
        
        if not args.is_multiview:
            train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True)
        else:
            train_set=Dataset_train_multiview(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
            print('Multiview training')
            args.views = [1,2,3,4]
            args.sample_rate = 16000
            args.padding_type = 'repeat'
            args.random_start = False
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True, collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start))
        
        del train_set, label_trn
        
        # define validation dataloader
        labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
        print('no. of validation trials',len(files_id_dev))

        
        if not args.is_multiview:
            dev_set = Dataset_train(args,list_IDs = files_id_dev,
                labels = labels_dev,
                base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False)
        else:
            dev_set = Dataset_train_multiview(args,list_IDs = files_id_dev, labels=labels_dev, base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False, collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start))
        del dev_set,labels_dev

        
        ##################### Training and validation #####################
        num_epochs = args.num_epochs
        not_improving=0
        epoch=0
        bests=np.ones(n_mejores,dtype=float)*float('inf')
        best_loss=float('inf')
        if args.train:
            for i in range(n_mejores):
                np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
            while not_improving<args.num_epochs:
                print('######## Epoca {} ########'.format(epoch))

                if not args.is_multiview:
                    train_epoch(train_loader, model, args.lr, optimizer, device)
                    val_loss = evaluate_accuracy(dev_loader, model, device)
                else:
                    weighted_views = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
                    train_epoch_multiview(train_loader, model, optimizer, device, weighted_views)
                    val_loss, _ = dev_epoch_multiview(dev_loader, model, device, weighted_views)

                if val_loss<best_loss:
                    best_loss=val_loss
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                    print('New best epoch')
                    not_improving=0
                else:
                    not_improving+=1
                for i in range(n_mejores):
                    if bests[i]>val_loss:
                        for t in range(n_mejores-1,i,-1):
                            bests[t]=bests[t-1]
                            os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                        bests[i]=val_loss
                        torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                        break
                print('\n{} - {}'.format(epoch, val_loss))
                print('n-best loss:', bests)
                #torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                epoch+=1
                if epoch>74:
                    break
            print('Total epochs: ' + str(epoch) +'\n')


        print('######## Eval ########')
        if args.average_model:
            sdl=[]
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1,args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key]=(sd[key]+sd2[key])
            for key in sd:
                sd[key]=(sd[key])/args.n_average_model
            model.load_state_dict(sd)
            torch.save(model.state_dict(), os.path.join(best_save_path, 'avg_5_best_{}.pth'.format(i)))
            print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

        eval_tracks=['DF']
        if args.comment_eval:
            model_tag = model_tag + '_{}'.format(args.comment_eval)

        for tracks in eval_tracks:
            if not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag)):
                prefix      = 'ASVspoof_{}'.format(tracks)
                prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
                prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

                file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format( prefix,prefix_2021)), is_eval=True)
                print('no. of eval trials',len(file_eval))
                eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)),track=tracks)
                produce_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, model_tag))
            else:
                print('Score file already exists')
    
    else:
        print(f'other dataset: {args.dataset}')
        label_trn, files_id_train = read_metadata_other( dir_meta =  os.path.join(args.protocols_path), is_train=True)
        print('no. of training trials',len(files_id_train))
        
        if not args.is_multiview:
            train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path),algo=args.algo, format='')
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True)
        else:
            train_set=Dataset_train_multiview(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path),algo=args.algo,format='')
            print('Multiview training')
            args.views = [1,2,3,4]
            args.sample_rate = 16000
            args.padding_type = 'repeat'
            args.random_start = False
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True, collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start))
        
        del train_set, label_trn
        
        # define validation dataloader
        labels_dev, files_id_dev = read_metadata_other( dir_meta =  os.path.join(args.protocols_path), is_dev=True)
        print('no. of validation trials',len(files_id_dev))

        
        if not args.is_multiview:
            dev_set = Dataset_train(args,list_IDs = files_id_dev,
                labels = labels_dev,
                base_dir = os.path.join(args.database_path), algo=args.algo, format='')
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False)
        else:
            dev_set = Dataset_train_multiview(args,list_IDs = files_id_dev, labels=labels_dev, base_dir = os.path.join(args.database_path), algo=args.algo, format='')
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False, collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start))
        del dev_set,labels_dev

        num_epochs = args.num_epochs
        not_improving=0
        epoch=0
        bests=np.ones(n_mejores,dtype=float)*float('inf')
        best_loss=float('inf')
        if args.train:
            for i in range(n_mejores):
                np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
            while not_improving<args.num_epochs:
                print('######## Epoca {} ########'.format(epoch))
                if not args.is_multiview:
                    train_epoch(train_loader, model, args.lr, optimizer, device)
                    val_loss = evaluate_accuracy(dev_loader, model, device)
                else:
                    weighted_views = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
                    train_epoch_multiview(train_loader, model, optimizer, device, weighted_views)
                    val_loss, _ = dev_epoch_multiview(dev_loader, model, device, weighted_views)
                    
                if val_loss<best_loss:
                    best_loss=val_loss
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                    print('New best epoch')
                    not_improving=0
                else:
                    not_improving+=1
                for i in range(n_mejores):
                    if bests[i]>val_loss:
                        for t in range(n_mejores-1,i,-1):
                            bests[t]=bests[t-1]
                            os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                        bests[i]=val_loss
                        torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                        break
                print('\n{} - {}'.format(epoch, val_loss))
                print('n-best loss:', bests)
                #torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                epoch+=1
                if epoch>74:
                    break
            print('Total epochs: ' + str(epoch) +'\n')


        print('######## Eval ########')
        if args.average_model:
            sdl=[]
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1,args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key]=(sd[key]+sd2[key])
            for key in sd:
                sd[key]=(sd[key])/args.n_average_model
            model.load_state_dict(sd)
            torch.save(model.state_dict(), os.path.join(best_save_path, 'avg_5_best_{}.pth'.format(i)))
            print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

        eval_tracks=['LA', 'DF']
        if args.comment_eval:
            model_tag = model_tag + '_{}'.format(args.comment_eval)

        file_eval, _ = read_metadata_other( dir_meta = args.protocols_path, is_eval=True)
        print('no. of eval trials',len(file_eval))
        if args.var:
            print('var-length eval')
            eval_set=Dataset_var_eval2(list_IDs = file_eval,base_dir = args.database_path, format='')
            produce_evaluation_file(eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
        else:
            eval_set=Dataset_eval(list_IDs = file_eval,base_dir = args.database_path, cut=args.cut, track='',format='')
            produce_evaluation_file(eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
