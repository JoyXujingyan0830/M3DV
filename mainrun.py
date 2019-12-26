#!/usr/bin/env python

import logging
import os
import sys
import argparse
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
#from myModel import generate_model
#import resnet

#from dataset import create_dataloader_train_val
#from model import toy_cnn
from torchvision import transforms

np.random.seed(1)
torch.manual_seed(1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,) * 100, std=(0.5,) * 100)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, net, loss_fn, optimizer):
    net.train()

    total_loss = []
    total_output = []
    total_label = []
    with torch.set_grad_enabled(True):
        for X, y in dataloader:
            X = X.float().unsqueeze(1).to(device)
            y = y.long().to(device)
            out = net(X).to(device)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

            total_output.extend(out.argmax(1).tolist())
            total_label.extend(y.tolist())

    return np.mean(total_loss), np.sum(np.array(total_output) == np.array(total_label)) / len(total_output)


def val_epoch(dataloader, net, loss_fn):
    net.eval()

    total_loss = []
    total_output = []
    total_label = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.float().unsqueeze(1).to(device)
            y = y.long().to(device)
            out = net(X)
            loss = loss_fn(out, y)

            total_loss.append(loss.item())

            total_output.extend(out.argmax(1).tolist())
            total_label.extend(y.tolist())

    return np.mean(total_loss), np.sum(np.array(total_output) == np.array(total_label)) / len(total_output)


def predict_test(net, test_dir, out_file):
    writer = csv.writer(open(out_file, "w", encoding="utf-8"))
    writer.writerow(["Id", "Predicted"])
    test_list = os.listdir(test_dir)
    test_list = sorted(test_list, key=lambda x: int(x.split(".")[0][9:]))
    for test_input in test_list:
        x_test = np.load(os.path.join(test_dir, test_input))["voxel"]
        x_test = transform(x_test)
        x_test = x_test.unsqueeze(0).unsqueeze(0)
        x_test = x_test.float().to(device)
        prob = torch.softmax(net(x_test), dim=1)[0, 1].item()
        writer.writerow([test_input.split(".")[0], prob])


def getLogger(outputfile):
    formatter = logging.Formatter("[ %(levelname)s ]\t%(message)s")
    logging.basicConfig(filemode="w", level=logging.INFO)
    logger = logging.getLogger(outputfile)
    fh = logging.FileHandler(outputfile)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,) * 100, std=(0.5,) * 100)
])



class toy_cnn(nn.Module):

    def __init__(self):
        super(toy_cnn, self).__init__()
        self.conv1 = nn.Sequential(
        	#nn.BatchNorm3d(1),
        	#nn.ReLU(),
            nn.Conv3d(1, 16, 5, 1, 2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2))
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(512, 1024, 3, 1, 1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.outputlayer = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        #print ('conv1 done')
        x = self.conv2(x)
        #print ('conv2 done')
        x = self.conv3(x)
        #print ('conv3 done')
        x = self.conv4(x)
       # print ('conv4 done')
      #  x = self.conv5(x)
       # print ('conv5 done')
      #  x = self.conv6(x)
     #   x = self.conv7(x)

        x = self.pooling(x)
        #print ('pooling done')
        x = x.reshape(x.size(0), x.size(1))
       # print ('reshape done')
        return self.outputlayer(x)



class m3dv_dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, label_df):
        super(m3dv_dataset, self).__init__()
        self._data_dir = data_dir
        self._label_df = label_df


    def __getitem__(self, index):
        name = self._label_df.iloc[index]["name"]
        data = np.load(os.path.join(self._data_dir, "{}.npz".format(name)))
        voxel = data["voxel"]
        seg = data["seg"]
        label = self._label_df.iloc[index]["label"]
        feature = voxel
        feature = transform(feature)
        return feature, label

    def __len__(self):
        return len(self._label_df)


def create_dataloader(data_dir,
                      label_df,
                      batch_size,
                      shuffle,
                      num_workers):
    ds = m3dv_dataset(data_dir, label_df)
    return torch.utils.data.DataLoader(ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def create_dataloader_train_val(data_dir,
                                label_df,
                                batch_size,
                                num_workers,
                                percent):
    train_df = label_df.sample(frac=percent / 100., random_state=0)
    val_df = label_df[~label_df.index.isin(train_df.index)]
    return create_dataloader(data_dir,
                             train_df,
                             batch_size,
                             True,
                             num_workers), \
        create_dataloader(data_dir,
                          val_df,
                          batch_size,
                          False,
                          num_workers)
'''
class EarlyStopping(object):

    def __init__(self, patience, score_function, trainer):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        self.score_function = score_function
        self.patience = patience
        self.trainer = trainer
        self.counter = 0
        self.best_score = None
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            self._logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self._logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_score = score
            self.counter = 0
'''

def main():

    if not os.path.exists("exp"):
        os.mkdir("exp")
    logfile = os.path.join("exp", "train.log")

    train_val_data_dir = "train_val"
    train_val_label_df = pd.read_csv("train_val.csv")
    train_dataloader, val_dataloader = create_dataloader_train_val(
        train_val_data_dir, 
        train_val_label_df,
        batch_size=25,
        num_workers=16,
        percent=75)
    print ('data fine')
    
  #  model,parameters = generate_model(models='resnet', model_depth=50, n_classes=2, shortcut='B', sample_size=100, sample_duration=100, pretrain_path=False, ft_begin_index=0)

   # optimizer = torch.optim.Adam(parameters, lr=learning_rate)

   # criterion = nn.CrossEntropyLoss(size_average=True)

    #for epoch in range(num_epoches):
    #    print('epoch:'+str(epoch))
    #    train(train_loader, model, criterion, optimizer, epoch)

    net = toy_cnn().to(device)
    '''
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=1e-6)

    logger = getLogger(logfile)
    logger.info("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc")

    best_loss = np.inf
    best_acc = -1
    print ('net fine')
    #EarlyStopping(patience = 0.1, score_function = val_acc, trainer =train_epoch(train_dataloader, net, loss_fn, optimizer) )
    for epoch in range(1, 80 + 1):
        print (epoch)
        train_loss, train_acc = train_epoch(train_dataloader, net, loss_fn, optimizer)
        val_loss, val_acc = val_epoch(
            val_dataloader, net, loss_fn)
        logger.info("{}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}".format(logger, train_loss, train_acc, val_loss, val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net, os.path.join("exp", "model2.pth"))
        if val_acc > best_acc:
            best_acc = val_acc

    logger.info("best acc: {:.2f}".format(best_acc))
'''    
net = torch.load("model2.pth").to(device)
predict_test(net, "test/", "Submission.csv")
print ('already written')

main()
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="exp")
    parser.add_argument("--num-epochs", type=int, default=20)
    # parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--out-file", type=str, default="Submission.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-percent", type=int, default=75)

    args = parser.parse_args()
    main(args)
'''