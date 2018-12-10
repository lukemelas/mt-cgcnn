import argparse
import sys
import os
import shutil
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from model import CrystalGraphConvNet
from data import collate_pool, get_train_val_test_loader, CIFData
from utils import Normalizer, mae, class_eval, AverageMeter, save_checkpoint 

parser = argparse.ArgumentParser(description='CGCNN')

# Input and output
parser.add_argument('--no_gpu', action='store_true', help='cpu')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
    help='dataset options, starting with the path to root dir') 
parser.add_argument('--task', choices=['regression', 'classification'],
    default='regression', help='type of prediction task')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
    help='number of data loading workers (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument('--train-size', default=None, type=int, metavar='N',
    help='number of training data to be loaded (default none)')
parser.add_argument('--val-size', default=1000, type=int, metavar='N',
    help='number of validation data to be loaded (default 1000)')
parser.add_argument('--test-size', default=1000, type=int, metavar='N',
    help='number of test data to be loaded (default 1000)')

# Optimization
parser.add_argument('--epochs', default=30, type=int, metavar='N',
    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
    metavar='N', help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
    help='choose an optimizer, SGD or Adam, (default: SGD)')

# Model
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
    help='number of hidden atom features in conv layers')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
    help='number of conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
    help='number of hidden features after pooling')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
    help='number of hidden layers after pooling')

# Parse
args = parser.parse_args(sys.argv[1:])
args.cuda = not args.no_gpu and torch.cuda.is_available()
print('GPU ' + ('enabled' if args.cuda else 'disabled'))

# Initialize best error
if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

def main():
    global args, best_mae_error

    # Dataset from CIF files
    dataset = CIFData(*args.data_options)
    print(f'Dataset size: {len(dataset)}')

    # Dataloader from dataset
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, 
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_size=args.train_size, 
        num_workers=args.workers,
        val_size=args.val_size, 
        test_size=args.test_size,
        pin_memory=args.cuda, 
        return_test=True
    )

    # Initialize data normalizer with sample of 500 points
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    elif args.task == 'regression':
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)
    else:
        raise NameError('task argument must be regression or classification')

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len,
                                nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=(args.task=='classification'))

    # GPU
    if args.cuda:
        model.cuda()

    # Loss function 
    criterion = nn.NLLLoss() if args.task == 'classification' else nn.MSELoss()

    # Optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('optim argument must be SGD or Adam')

    # Scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # Resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error   = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train
    for epoch in range(args.start_epoch, args.epochs):

        # Train (one epoch)
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # Validate 
        mae_error = validate(val_loader, model, criterion, normalizer)
        assert mae_error == mae_error, 'NaN :('

        # Step learning rate scheduler
        scheduler.step(mae_error)

        # Save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # Evaluate best model on test set
    print('--------- Evaluate model on test set ---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    '''Train for a single epoch'''
    model.train()

    # Statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # Loop through minibatches
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # GPU
        if args.cuda:
            input_var = (Variable(input[0].cuda(async=True)),
                         Variable(input[1].cuda(async=True)),
                         input[2].cuda(async=True),
                         [crys_idx.cuda(async=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])

        # Normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(async=True))
        else:
            target_var = Variable(target_normed)

        # Forward 
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy 
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            tmp = class_eval(output.data.cpu(), target)
            accuracy, precision, recall, fscore, auc_score = tmp
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, mae_errors=mae_errors)
                      )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, accu=accuracies,
                       prec=precisions, recall=recalls, f1=fscores,
                       auc=auc_scores)
                      )


def validate(val_loader, model, criterion, normalizer, test=False):
    '''Test on validation/test set'''
    model.eval()

    # Statistics
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # Minibatches
    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):

        # GPU
        if args.cuda:
            input_var = (Variable(input[0].cuda(async=True), volatile=True),
                         Variable(input[1].cuda(async=True), volatile=True),
                         input[2].cuda(async=True),
                         [crys_idx.cuda(async=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0], volatile=True),
                         Variable(input[1], volatile=True),
                         input[2],
                         input[3])

        # Normalize
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(async=True), volatile=True)
        else:
            target_var = Variable(target_normed, volatile=True)

        # Forward
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Accuracy
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # Time 
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg
if __name__ == '__main__':
    main()
