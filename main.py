

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import progress_bar, visualize_tsne
from networks.resnet_big import SupLLMResNet
from losses import SupLLMLoss
from caption_dataset import CaptionDataset, labels_to_words_cifar10, TextCoder
from text_autoencoders.model import AAE
from text_autoencoders.vocab import Vocab
from text_autoencoders.utils import strip_eos
from text_autoencoders.model import reparameterize
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')

    parser.add_argument('--caption_model', type=str, default="git-large", choices=["blip-base", "blip-large", "git-base", "git-large"])

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()


    # set autoencoder options
    opt.lambda_kl = 0
    opt.lambda_adv = 10
    opt.lambda_p = 0
    opt.noise = "0.3,0,0,0"
    opt.dropout = 0.5
    opt.dim_z = 128
    opt.dim_emb = 512
    opt.dim_h = 1024
    opt.nlayers = 1
    opt.dim_d = 512
    opt.lr = 0.0005
    opt.lambda_adv = 1

    # set the path according to the environment
    opt.data_folder = '/home/pplcount/users/ottj/data/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    opt.daae_path = './text_autoencoders/checkpoints/yelp/daae/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
        opt.cosine = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    vocab_file = os.path.join(opt.daae_path, 'vocab.txt')
    vocab = Vocab(vocab_file)

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,                                         
                                         download=False)
        train_dataset = CaptionDataset(dataset=train_dataset, transform=train_transform, mode="train", dataset_name=opt.dataset, caption_model=opt.caption_model, vocab=vocab)

        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False)

        val_dataset = CaptionDataset(dataset=val_dataset, transform=val_transform, mode="val", dataset_name=opt.dataset, caption_model=opt.caption_model, vocab=vocab)

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          download=False)

        train_dataset = CaptionDataset(dataset=train_dataset, transform=train_transform, mode="train", dataset_name=opt.dataset, caption_model=opt.caption_model, vocab = vocab)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False)
        
        val_dataset = CaptionDataset(dataset=val_dataset, transform=val_transform, mode="val", dataset_name=opt.dataset, caption_model=opt.caption_model, vocab=vocab)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True)

    # visualize_tsne(val_loader)

    return train_loader, val_loader


def set_model(opt):
    model = SupLLMResNet(name=opt.model, feat_dim=128, num_classes=opt.n_cls)
    vocab_file = os.path.join(opt.daae_path, 'vocab.txt')
    vocab = Vocab(vocab_file)
    autoencoder = AAE(
        vocab, opt)
    
    criterion = torch.nn.NLLLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        autoencoder.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, autoencoder


def train(train_loader, model, autoencoder, criterion, optimizer, epoch, opt):
    """one epoch training"""
    print('\n Epoch: %d' % epoch)
    model.train()


    train_loss = 0
    train_loss_g = 0
    train_loss_d = 0
    correct = 0
    total = 0

    for idx, (images, labels, caption_input, caption_target) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        caption_input = caption_input.cuda(non_blocking=True)
        caption_target = caption_target.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output, feat_mu, feat_logvar = model(images)
        
        feat = reparameterize(feat_mu, feat_logvar)

        logits, _ = autoencoder.decode(feat, caption_input.t())
        losses = {}
        loss_rec = autoencoder.loss_rec(logits, caption_target.t()).mean()
        loss_d, loss_g = autoencoder.loss_adv(feat)
        losses['loss_d'] = loss_d

        losses["loss"] = loss_rec
        autoencoder.step(losses)
        
        loss = criterion(output,labels) + opt.lambda_adv * loss_g


        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        train_loss_d += loss_d.item()
        train_loss_g += loss_g.item()
        _ , predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(idx, len(train_loader), 'Loss: %.3f , Loss G: %.3f, Loss D %.3f| Acc: %.3f%% (%d/%d)'
                    % (train_loss/(idx+1), train_loss_g/(idx+1), train_loss_d/(idx+1), 100.*correct/total, correct, total))

    return train_loss / total , 100*correct/total


def validate(val_loader, model, autoencoder, criterion, opt, return_embeddings=False):
    """validation"""
    model.eval()

    batch_time = AverageMeter()

    val_loss = 0
    correct = 0
    total = 0
    if return_embeddings:
        embeddings = torch.zeros(size=(len(val_loader)*100, 768))

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, caption_input, caption_target) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            caption_input = caption_input.cuda(non_blocking=True)
            caption_target = caption_target.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output, feat, _ = model(images)
            
            loss = criterion(output, labels)

            # update metric
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (val_loss/(idx+1), 100.*correct/total, correct, total))
    
    #generate text from the last feature
    with torch.no_grad():
        text = autoencoder.generate(feat, max_len=15, alg="greedy").t()
    
    """
    For Text Generation
    sents = []
    for text in [text]:
        for s in text:
            sents.append([autoencoder.vocab.idx2word[id] for id in s[1:]])
    """
    
    if return_embeddings:
        return val_loss / total, 100* correct / total, embeddings

    return val_loss / total, 100* correct / total


def main():
    best_acc = 0
    opt = parse_option()

    print(opt)
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion, autoencoder = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss, train_acc = train(train_loader, model,autoencoder, criterion, optimizer, epoch, opt)

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, autoencoder, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, autoencoder, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, autoencoder, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


def main_val():
    opt = parse_option()
    save_file = os.path.join(
        opt.save_folder, 'last.pth')

if __name__ == '__main__':
    #main_val()
    main()
