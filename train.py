import argparse

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

from dataset import COCOdataset
from fast_rcnn import FastRCNN


def train(model, train_dataset, optimizer, args):
    model.train()
    num_batches = len(train_dataset) // args.batch_size
    indexes = np.random.shuffle(np.arange(len(train_dataset)))
    loss_all = 0
    loss_cat_all = 0
    loss_loc_all = 0
    accuracy = 0
    num_samples = 0
    for i in range(num_batches):
        imgs = []
        rects = []
        roi_idxs = []
        rela_locs = []
        cats = []
        for j in range(args.batch_size):
            img, rect, roi_idx_len, rela_loc, cat = train_dataset[i *
                                                                 args.batch_size+j]
            # print(img, rect, roi_idx_len, gt_rect, cat)
            imgs.append(img.unsqueeze(0))
            rects += rect
            rela_locs += rela_loc
            roi_idxs += ([j] * roi_idx_len)
            cats += cat
        imgs = torch.cat(imgs, dim=0)
        rects = np.array(rects)
        rela_locs = torch.FloatTensor(rela_locs)
        cats = torch.LongTensor(cats)
        # print(imgs, rects, roi_idxs, rela_locs, cats)
        if args.cuda:
            imgs = imgs.cuda()
            rela_locs = rela_locs.cuda()
            cats = cats.cuda()
        optimizer.zero_grad()
        prob, bbox = model.forward(imgs, rects, roi_idxs)
        loss, loss_cat, loss_loc = model.loss(prob, bbox, cats, rela_locs)
        loss.backward()
        optimizer.step()
        num_samples += len(cats)
        loss_all += loss.item() * len(cats)
        loss_cat_all += loss_cat.item() * len(cats)
        loss_loc_all += loss_loc.item() * len(cats)
        accuracy += (torch.argmax(prob.detach(), dim=-1) == cats).sum().item()
    return model, loss_all/num_samples, loss_cat_all/num_samples, loss_loc_all/num_samples, accuracy/num_samples


def test(model, val_dataset, args):
    model.eval()
    num_batches = len(val_dataset) // args.batch_size
    indexes = np.random.shuffle(np.arange(len(val_dataset)))
    loss_all = 0
    loss_cat_all = 0
    loss_loc_all = 0
    accuracy = 0
    num_samples = 0
    for i in range(num_batches):
        imgs = []
        rects = []
        roi_idxs = []
        rela_locs = []
        cats = []
        for j in range(args.batch_size):
            img, rect, roi_idx_len, rela_loc, cat = val_dataset[i *
                                                               args.batch_size+j]
            # print(img, rect, roi_idx_len, gt_rect, cat)
            imgs.append(img.unsqueeze(0))
            rects += rect
            rela_locs += rela_loc
            roi_idxs += ([j] * roi_idx_len)
            cats += cat
        imgs = torch.cat(imgs, dim=0)
        rects = np.array(rects)
        rela_locs = torch.FloatTensor(rela_locs)
        cats = torch.LongTensor(cats)
        # print(imgs, rects, roi_idxs, rela_locs, cats)
        if args.cuda:
            imgs = imgs.cuda()
            rela_locs = rela_locs.cuda()
            cats = cats.cuda()
        prob, bbox = model.forward(imgs, rects, roi_idxs)
        loss, loss_cat, loss_loc = model.loss(prob, bbox, cats, rela_locs)
        num_samples += len(cats)
        loss_all += loss.item() * len(cats)
        loss_cat_all += loss_cat.item() * len(cats)
        loss_loc_all += loss_loc.item() * len(cats)
        accuracy += (torch.argmax(prob.detach(), dim=-1) == cats).sum().item()
    return model, loss_all/num_samples, loss_cat_all/num_samples, loss_loc_all/num_samples, accuracy/num_samples


def main():
    parser = argparse.ArgumentParser('parser for fast-rcnn')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_path', type=str,
                        default='./model/fast_rcnn.pkl')
    parser.add_argument('--cuda', type=bool, default=True)

    args = parser.parse_args()
    train_dataset = COCOdataset()
    valid_dataset = COCOdataset(mode='val')
    model = FastRCNN(num_classes=args.num_classes)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print("Epoch %d:" % epoch)
        model, train_loss, train_loss_cat, train_loss_loc, train_accuracy = train(
            model, train_dataset, optimizer, args)
        print("Train: loss=%.4f, loss_cat=%.4f, loss_loc=%.4f, accuracy=%.4f" %
              (train_loss, train_loss_cat, train_loss_loc, train_accuracy))
        model, valid_loss, valid_loss_cat, valid_loss_loc, valid_accuracy = test(
            model, valid_dataset, args)
        print("Valid: loss=%.4f, loss_cat=%.4f, loss_loc=%.4f, accuracy=%.4f" %
              (valid_loss, valid_loss_cat, valid_loss_loc, valid_accuracy))

    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()
