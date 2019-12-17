import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from models.cnn import CNN

class Model():
    def __init__(self, device, dataloader, config):
        self.net = CNN().to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        lr = config['learning_rate']
        momentum = config['momentum']
        weight_decay = config['weight_decay']
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        step_size = config['step_size']
        lr_gamma = config['lr_gamma']
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size = step_size,
                                                   gamma = lr_gamma)
        self.dataloader = dataloader
        self.checkpointDir = config['save_path'] + 'checkpoint_%2d.pt'

    def loadModel(self, epoch):
        checkpoint = torch.load(self.checkpointDir %
                                (epoch), map_location=torch.device(self.device))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def train(self, config):
        epochs = config['epochs']
        resumeTraining = config['load_model']

        if(resumeTraining != 0):
            self.loadModel(resumeTraining)

        running_loss = 0
        iteration = 0
        for I in range(resumeTraining, epochs + resumeTraining, 1):
            self.net.train()
            for i, (data, labels) in enumerate(self.dataloader.training, 0):
                data = data.to(device=self.device, dtype=torch.float)
                labels = labels.to(device=self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                output = self.net(data).view(-1)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if (config['debug'] >= 1 and iteration % 30 == 29):
                    print('[%2d/%5d] loss: %.5f' %
                          (I + 1, iteration + 1, running_loss/30), flush=True)
                    running_loss = 0.0
                iteration = iteration + 1

            torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, self.checkpointDir % (I+1))
            self.scheduler.step()


    def validate(self, config):
        epoch = config['test_model_epoch']
        if(epoch == -1):
            print('Invalid testing epoch')
            return
        checkpoint = torch.load(self.checkpointDir % (epoch),
                                map_location=torch.device(self.device))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        correct = 0
        label = []
        predclass = []
        pred = []
        for j, (data, labels) in enumerate(self.dataloader.validating, 0):
            data = data.to(device=self.device, dtype=torch.float)
            output = self.net(data).view(-1)
            output_class = np.where(output.cpu() < 0.5, 0, 1)
            correct = correct + np.sum(output_class == labels.numpy())
            if(config['debug'] >= 1):
                print("Truth: {} / Prediction: [{}/{}]"
                      .format(labels.numpy()[0],
                              output.detach().cpu().numpy()[0],
                              output_class[0]), flush = True)
            label.extend(labels.numpy())
            predclass.extend(output_class)
            pred.extend(output.detach().cpu().numpy())
        print("Correct classictions: {%d} / {%d} " %
              (correct, self.dataloader.test_size))
        print(confusion_matrix(np.array(label), np.array(predclass)))
        print(roc_auc_score(np.array(label), np.array(pred)))
