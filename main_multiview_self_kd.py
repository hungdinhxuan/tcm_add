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
from losses import loss_fn_kd
from utils import ExperimentLogger

import argparse
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    """
    Parse command line arguments. Now only requires a path to the config file.
    
    Returns:
        Namespace object containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Load and merge YAML config with command line arguments
    config = load_config(args.config)
    
    # Convert dictionary to Namespace for compatibility
    args = argparse.Namespace(**config)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')
        
    return args

def calculate_cascade_kd_loss(logits_view_dict, alpha, T):
    """
    Calculate cascading knowledge distillation loss where higher views teach lower views.
    For example: view 4 teaches view 3, view 3 teaches view 2, view 2 teaches view 1.
    
    Args:
        logits_view_dict: Dictionary containing logits for each view
        alpha: Weight for balancing soft and hard targets
        T: Temperature parameter
        
    Returns:
        Dictionary of KD losses for each student view and total KD loss
    """
    kd_losses = {}
    total_kd_loss = 0.0
    
    # Sort views in descending order to ensure proper teaching cascade
    views = sorted([int(k) for k in logits_view_dict.keys()], reverse=True)
    
    # Calculate KD loss for each student-teacher pair
    for i in range(len(views)-1):
        teacher_view = str(views[i])     # Higher view (teacher)
        student_view = str(views[i+1])   # Lower view (student)

        # import sys
        # print(f"Teacher view: {teacher_view}, Student view: {student_view}")
        # sys.exit(1)
        
        # Get logits for current teacher-student pair
        teacher_logits = logits_view_dict[teacher_view]
        student_logits = logits_view_dict[student_view]
        
        # Calculate KD loss
        kd_loss = loss_fn_kd(student_logits, teacher_logits, alpha, T)
        
        # Store individual loss and add to total
        kd_losses[f'view_{student_view}'] = kd_loss
        total_kd_loss += kd_loss
    
    return kd_losses, total_kd_loss

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i = 0
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
            i = i+1
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('\n{} - {} - {} '.format(epoch, str(test_accuracy)+'%', val_loss))
    return val_loss

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=64,
                             shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []

    with torch.no_grad():
        for batch_x, utt_id in data_loader:
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

def train_epoch(train_loader, model, lr, optim, device):
    num_total = 0.0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i = 0
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        batch_x, batch_y, utt_id = batch

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
        i = i+1
    sys.stdout.flush()

def train_epoch_multiview(
    train_loader,
    model,
    optimizer,
    device,
    logger: ExperimentLogger,
    weighted_views: Dict[str, float] = {
        '1': 0.9, '2': 0.9, '3': 0.9, '4': 0.9},
    ce_weight=[0.1, 0.9],
    T=4,
    alpha=0.1
):
    model.train()
    weight = torch.FloatTensor(ce_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Initialize metrics for each view
    running_losses = {f'view_{v}': 0.0 for v in weighted_views.keys()}
    running_corrects = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_samples = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_kd_losses = {f'view_{v}': 0.0 for v in weighted_views.keys() if v != '4'}
    total_running_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        batch_total_loss = 0.0
        logits_view_dict = {}

        # Process each view
        for view_idx, (inputs, labels) in batch.items():
            view = str(view_idx)  # Convert view index to string

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.view(-1).type(torch.int64).to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs, _ = model(inputs)

            logits_view_dict[view] = outputs

            loss = criterion(outputs, labels) * weighted_views[view]

            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()

            # Update running metrics
            running_losses[f'view_{view}'] += loss.item() * batch_size
            running_corrects[f'view_{view}'] += correct
            running_samples[f'view_{view}'] += batch_size

            # Accumulate total loss
            batch_total_loss += loss


        '''
            [Calculate KD loss for each view, higher view will teach lower view]
            For each example view 4 will teach view 3, view 3 will teach view 2, and view 2 will teach view 1.
            The KD loss is calculated as the difference between the softmax outputs of the teacher and student models.
            The KD loss is added to the total loss.
        '''
        # Calculate cascading knowledge distillation losses
        kd_losses, kd_total_loss = calculate_cascade_kd_loss(logits_view_dict, alpha, T)

        # Add KD losses to the total loss
        batch_total_loss += kd_total_loss

        # Update running KD losses
        for view, loss in kd_losses.items():
            running_kd_losses[view] += loss.item() * batch_size

        # Update total running loss
        total_running_loss += batch_total_loss.item() * batch_size
        total_samples += batch_size
                    
        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        batch_total_loss.backward()
        optimizer.step()

        # Update progress bar with accumulated metrics
        desc_parts = [
            f"Total Loss: {total_running_loss/total_samples:.4f}"
        ]

        # Calculate step for logging
        global_step = epoch * len(train_loader) + batch_idx
        
        # Log batch metrics
        batch_metrics = {
            'batch/total_loss': batch_total_loss.item(),
            'batch/kd_loss': kd_total_loss.item()
        }
        
        # Add CE losses
        for view in weighted_views.keys():
            view_samples = running_samples[f'view_{view}']
            if view_samples > 0:
                desc_parts.append(
                    f"V{view} Loss: {running_losses[f'view_{view}']/view_samples:.4f}"
                )
        
        # Add KD losses
        for view in weighted_views.keys():
            if view != '4' and f'view_{view}' in running_kd_losses:
                view_samples = running_samples[f'view_{view}']
                if view_samples > 0:
                    desc_parts.append(
                        f"V{view} KD: {running_kd_losses[f'view_{view}']/view_samples:.4f}"
                    )
        
        # Add accuracies
        for view in weighted_views.keys():
            view_samples = running_samples[f'view_{view}']
            if view_samples > 0:
                desc_parts.append(
                    f"V{view} Acc: {running_corrects[f'view_{view}']/view_samples:.4f}"
                )
        # Log metrics
        #logger.log_metrics(batch_metrics, global_step)

        pbar.set_description(" | ".join(desc_parts))

    # Calculate epoch metrics
    epoch_metrics = {}
    for view in weighted_views.keys():
        view_key = f'view_{view}'
        if running_samples[view_key] > 0:
            epoch_metrics[f'train_loss/{view_key}'] = running_losses[view_key] / \
                running_samples[view_key]
            epoch_metrics[f'train_acc/{view_key}'] = running_corrects[view_key] / \
                running_samples[view_key]
            
            # Add KD loss metrics for views 1-3
            if view != '4' and view_key in running_kd_losses:
                epoch_metrics[f'train_loss_kd/{view_key}'] = running_kd_losses[view_key] / \
                    running_samples[view_key]
    # Log epoch metrics
    logger.log_metrics(epoch_metrics, epoch)
    return epoch_metrics

def dev_epoch_multiview(
    dev_loader,
    model,
    device,
    logger: ExperimentLogger,
    weighted_views: Dict[str, float] = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0},
    ce_weight=[0.1, 0.9],
    T=4,
    alpha=0.1
):
    model.eval()

    weight = torch.FloatTensor(ce_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Initialize metrics for each view
    running_losses = {f'view_{v}': 0.0 for v in weighted_views.keys()}
    running_corrects = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_samples = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_kd_losses = {f'view_{v}': 0.0 for v in weighted_views.keys() if v != '4'}
    total_running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dev_loader, desc='Validation')
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            batch_total_loss = 0.0
            logits_view_dict = {}

            # Calculate step for logging
            global_step = epoch * len(train_loader) + batch_idx

            # Process each view
            for view_idx, (inputs, labels) in batch.items():
                view = str(view_idx)  # Convert view index to string

                # Move data to device
                inputs = inputs.to(device)
                labels = labels.view(-1).type(torch.int64).to(device)
                batch_size = inputs.size(0)

                # Forward pass
                outputs, _ = model(inputs)
                logits_view_dict[view] = outputs

                loss = criterion(outputs, labels) * weighted_views[view]

                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels).sum().item()

                # Update running metrics
                running_losses[f'view_{view}'] += loss.item() * batch_size
                running_corrects[f'view_{view}'] += correct
                running_samples[f'view_{view}'] += batch_size

                # Accumulate total loss
                batch_total_loss += loss
            

            '''
            [Calculate KD loss for each view, higher view will teach lower view]
            For each example view 4 will teach view 3, view 3 will teach view 2, and view 2 will teach view 1.
            The KD loss is calculated as the difference between the softmax outputs of the teacher and student models.
            The KD loss is added to the total loss.
            '''
            # Calculate cascading knowledge distillation losses
            kd_losses, kd_total_loss = calculate_cascade_kd_loss(logits_view_dict, alpha, T)

            # Add KD losses to the total loss
            batch_total_loss += kd_total_loss

            

            # Update running KD losses
            for view, loss in kd_losses.items():
                running_kd_losses[view] += loss.item() * batch_size

            # Update total running loss
            total_running_loss += batch_total_loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar with accumulated metrics
            desc_parts = [
                f"Total Loss: {total_running_loss/total_samples:.4f}"
            ]


            # Calculate step for logging
            global_step = epoch * len(train_loader) + batch_idx
            
            # Log batch metrics
            batch_metrics = {
                'batch/val_total_loss': batch_total_loss.item(),
                'batch/val_kd_loss': kd_total_loss.item()
            }

            # Add CE losses
            for view in weighted_views.keys():
                view_samples = running_samples[f'view_{view}']
                if view_samples > 0:
                    desc_parts.append(
                        f"V{view} Loss: {running_losses[f'view_{view}']/view_samples:.4f}"
                    )

            # Add KD losses
            for view in weighted_views.keys():
                if view != '4' and f'view_{view}' in running_kd_losses:
                    view_samples = running_samples[f'view_{view}']
                    if view_samples > 0:
                        desc_parts.append(
                            f"V{view} KD: {running_kd_losses[f'view_{view}']/view_samples:.4f}"
                        )

            # Add accuracies
            for view in weighted_views.keys():
                view_samples = running_samples[f'view_{view}']
                if view_samples > 0:
                    desc_parts.append(
                        f"V{view} Acc: {running_corrects[f'view_{view}']/view_samples:.4f}"
                    )
            # Log metrics
            #logger.log_metrics(batch_metrics, global_step)

            pbar.set_description(" | ".join(desc_parts))

    # Calculate epoch metrics
    epoch_metrics = {}
    for view in weighted_views.keys():
        view_key = f'view_{view}'
        if running_samples[view_key] > 0:
            epoch_metrics[f'dev_loss/{view_key}'] = running_losses[view_key] / running_samples[view_key]
            epoch_metrics[f'dev_acc/{view_key}'] = running_corrects[view_key] / running_samples[view_key]

            # Add KD loss metrics for views 1-3
            if view != '4' and view_key in running_kd_losses:
                epoch_metrics[f'dev_loss_kd/{view_key}'] = running_kd_losses[view_key] / running_samples[view_key]

    # Calculate and return average loss for early stopping
    # Log epoch metrics
    logger.log_metrics(epoch_metrics, epoch)
    avg_loss = total_running_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss, epoch_metrics

if __name__ == '__main__':
    args = parse_arguments()    
    print(args)
    args.track = 'LA'

    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=f"self-KD-multiview-{args.comment}",
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "model_architecture": "Conformer-W2V-self-KD",
            "weighted_views": args.weighted_views,
            "temperature": args.T,
            "alpha": args.alpha
        }
    )

    # make experiment reproducible
    reproducibility(args.seed, args)

    track = args.track
    n_mejores = args.n_mejores_loss

    assert track in ['LA', 'DF'], 'Invalid track given'
    assert args.n_average_model < args.n_mejores_loss + \
        1, 'average models must be smaller or equal to number of saved epochs'

    # database
    prefix = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    # define model saving path
    model_tag = 'Conformer_self_MDT_KD_w_TCM_{}_{}_{}_ES{}_H{}_NE{}_KS{}_AUG{}_w_sin_pos'.format(
        track, args.loss, args.lr, args.emb_size, args.heads, args.num_encoders, args.kernel_size, args.algo)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    print('Model tag: ' + model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = Model(args, device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    nb_params = sum([param.view(-1).size()[0]
                    for param in model.parameters() if param.requires_grad])
    model = model.to(device)
    print('nb_params:', nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dataset == 'asvspoof':
        # define train dataloader
        label_trn, files_id_train = read_metadata(dir_meta=os.path.join(
            args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix, prefix_2019)), is_eval=False)
        print('no. of training trials', len(files_id_train))

        if not args.is_multiview:
            train_set = Dataset_train(args, list_IDs=files_id_train, labels=label_trn, base_dir=os.path.join(
                args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0], args.track)), algo=args.algo)
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)
        else:
            train_set = Dataset_train_multiview(args, list_IDs=files_id_train, labels=label_trn, base_dir=os.path.join(
                args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0], args.track)), algo=args.algo)
            print('Multiview training')
            args.views = [1, 2, 3, 4]
            args.sample_rate = 16000
            args.padding_type = 'repeat'
            args.random_start = args.random_start
            if args.random_start:
                print('Random start training')
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True,
                                      collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start, args.view_padding_configs))

        del train_set, label_trn

        # define validation dataloader
        labels_dev, files_id_dev = read_metadata(dir_meta=os.path.join(
            args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix, prefix_2019)), is_eval=False)
        print('no. of validation trials', len(files_id_dev))

        if not args.is_multiview:
            dev_set = Dataset_train(args, list_IDs=files_id_dev,
                                    labels=labels_dev,
                                    base_dir=os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0], args.track)), algo=args.algo)
            dev_loader = DataLoader(
                dev_set, batch_size=8, num_workers=10, shuffle=False)
        else:
            dev_set = Dataset_train_multiview(args, list_IDs=files_id_dev, labels=labels_dev, base_dir=os.path.join(
                args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0], args.track)), algo=args.algo)
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False, collate_fn=lambda x: multi_view_collate_fn(
                x, args.views, args.sample_rate, args.padding_type, args.random_start, args.view_padding_configs))
        del dev_set, labels_dev

        ##################### Training and validation #####################
        num_epochs = args.num_epochs
        not_improving = 0
        epoch = 0
        bests = np.ones(n_mejores, dtype=float)*float('inf')
        best_loss = float('inf')
        if args.train:
            for i in range(n_mejores):
                np.savetxt(os.path.join(best_save_path,
                           'best_{}.pth'.format(i)), np.array((0, 0)))
            while not_improving < args.num_epochs:
                print('######## Epoca {} ########'.format(epoch))

                if not args.is_multiview:
                    train_epoch(train_loader, model,
                                args.lr, optimizer, device)
                    val_loss = evaluate_accuracy(dev_loader, model, device)
                else:
                    weighted_views = args.weighted_views
                    train_epoch_multiview(
                        train_loader, model, optimizer, device, logger, weighted_views, ce_weight=[0.1, 0.9], T=args.T, alpha=args.alpha)
                    val_loss, _ = dev_epoch_multiview(
                        dev_loader, model, device, logger ,weighted_views, ce_weight=[0.1, 0.9], T=args.T, alpha=args.alpha)

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(
                        model_save_path, 'best.pth'))
                    print('New best epoch')
                    not_improving = 0
                else:
                    not_improving += 1
                for i in range(n_mejores):
                    if bests[i] > val_loss:
                        for t in range(n_mejores-1, i, -1):
                            bests[t] = bests[t-1]
                            os.system(
                                'mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                        bests[i] = val_loss
                        torch.save(model.state_dict(), os.path.join(
                            best_save_path, 'best_{}.pth'.format(i)))
                        break
                print('\n{} - {}'.format(epoch, val_loss))
                print('n-best loss:', bests)
                # torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                epoch += 1
                if epoch > 74:
                    break
            print('Total epochs: ' + str(epoch) + '\n')

        print('######## Eval ########')
        if args.average_model:
            sdl = []
            model.load_state_dict(torch.load(os.path.join(
                best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(
                best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1, args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(
                    best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(
                    best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key] = (sd[key]+sd2[key])
            for key in sd:
                sd[key] = (sd[key])/args.n_average_model
            model.load_state_dict(sd)
            torch.save(model.state_dict(), os.path.join(
                best_save_path, 'avg_5_best_{}.pth'.format(i)))
            print('Model loaded average of {} best models in {}'.format(
                args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(
                os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(
                os.path.join(model_save_path, 'best.pth')))

        eval_tracks = ['DF']
        if args.comment_eval:
            model_tag = model_tag + '_{}'.format(args.comment_eval)

        for tracks in eval_tracks:
            if not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag)):
                prefix = 'ASVspoof_{}'.format(tracks)
                prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
                prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

                file_eval = read_metadata(dir_meta=os.path.join(
                    args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_eval=True)
                print('no. of eval trials', len(file_eval))
                eval_set = Dataset_eval(list_IDs=file_eval, base_dir=os.path.join(
                    args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)), track=tracks)
                produce_evaluation_file(
                    eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, model_tag))
            else:
                print('Score file already exists')

    else:
        print(f'other dataset: {args.dataset}')
        files_id_train, label_trn = read_metadata_other(
            dir_meta=os.path.join(args.protocols_path), is_train=True)
        print('no. of training trials', len(files_id_train))

        if not args.is_multiview:
            train_set = Dataset_train(args, list_IDs=files_id_train, labels=label_trn, base_dir=os.path.join(
                args.database_path), algo=args.algo, format='')
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)
        else:
            train_set = Dataset_train_multiview(args, list_IDs=files_id_train, labels=label_trn, base_dir=os.path.join(
                args.database_path), algo=args.algo, format='')
            print('Multiview training')
            args.views = [1, 2, 3, 4]
            args.sample_rate = 16000
            args.padding_type = 'repeat'
            args.random_start = args.random_start
            if args.random_start:
                print('Random start training')
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True,
                                      collate_fn=lambda x: multi_view_collate_fn(x, args.views, args.sample_rate, args.padding_type, args.random_start))

        del train_set, label_trn

        # define validation dataloader
        files_id_dev, labels_dev = read_metadata_other(
            dir_meta=os.path.join(args.protocols_path), is_dev=True)
        print('no. of validation trials', len(files_id_dev))

        if not args.is_multiview:
            dev_set = Dataset_train(args, list_IDs=files_id_dev,
                                    labels=labels_dev,
                                    base_dir=os.path.join(args.database_path), algo=args.algo, format='')
            dev_loader = DataLoader(
                dev_set, batch_size=8, num_workers=10, shuffle=False)
        else:
            dev_set = Dataset_train_multiview(args, list_IDs=files_id_dev, labels=labels_dev, base_dir=os.path.join(
                args.database_path), algo=args.algo, format='')
            dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False, collate_fn=lambda x: multi_view_collate_fn(
                x, args.views, args.sample_rate, args.padding_type, args.random_start))
        del dev_set, labels_dev

        num_epochs = args.num_epochs
        not_improving = 0
        epoch = 0
        bests = np.ones(n_mejores, dtype=float)*float('inf')
        best_loss = float('inf')
        if args.train:
            for i in range(n_mejores):
                np.savetxt(os.path.join(best_save_path,
                           'best_{}.pth'.format(i)), np.array((0, 0)))
            while not_improving < args.num_epochs:
                print('######## Epoca {} ########'.format(epoch))
                if not args.is_multiview:
                    train_epoch(train_loader, model,
                                args.lr, optimizer, device)
                    val_loss = evaluate_accuracy(dev_loader, model, device)
                else:
                    weighted_views = args.weighted_views
                    # balanced weights for cross-entropy loss
                    ce_weight = [0.3, 0.7]
                    train_epoch_multiview(
                        train_loader, model, optimizer, device, logger, weighted_views, ce_weight, T=args.T, alpha=args.alpha)
                    val_loss, _ = dev_epoch_multiview(
                        dev_loader, model, device, logger, weighted_views, ce_weight, T=args.T, alpha=args.alpha)

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(
                        model_save_path, 'best.pth'))
                    print('New best epoch')
                    not_improving = 0
                else:
                    not_improving += 1
                for i in range(n_mejores):
                    if bests[i] > val_loss:
                        for t in range(n_mejores-1, i, -1):
                            bests[t] = bests[t-1]
                            os.system(
                                'mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                        bests[i] = val_loss
                        torch.save(model.state_dict(), os.path.join(
                            best_save_path, 'best_{}.pth'.format(i)))
                        break
                print('\n{} - {}'.format(epoch, val_loss))
                print('n-best loss:', bests)
                # torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                epoch += 1
                if epoch > 74:
                    break
            print('Total epochs: ' + str(epoch) + '\n')

        print('######## Eval ########')
        if args.average_model:
            sdl = []
            model.load_state_dict(torch.load(os.path.join(
                best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(
                best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1, args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(
                    best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(
                    best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key] = (sd[key]+sd2[key])
            for key in sd:
                sd[key] = (sd[key])/args.n_average_model
            model.load_state_dict(sd)
            torch.save(model.state_dict(), os.path.join(
                best_save_path, 'avg_5_best_{}.pth'.format(i)))
            print('Model loaded average of {} best models in {}'.format(
                args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(
                os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(
                os.path.join(model_save_path, 'best.pth')))

        eval_tracks = ['LA', 'DF']
        if args.comment_eval:
            model_tag = model_tag + '_{}'.format(args.comment_eval)

        file_eval, _ = read_metadata_other(
            dir_meta=args.protocols_path, is_eval=True)
        print('no. of eval trials', len(file_eval))
        if args.var:
            print('var-length eval')
            eval_set = Dataset_var_eval2(
                list_IDs=file_eval, base_dir=args.database_path, format='')
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
        else:
            eval_set = Dataset_eval(
                list_IDs=file_eval, base_dir=args.database_path, cut=args.cut, track='', format='')
            produce_evaluation_file(
                eval_set, model, device, 'Scores/{}.txt'.format(model_tag), args.batch_size)
