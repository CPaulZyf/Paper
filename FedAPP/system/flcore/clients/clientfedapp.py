import copy
import torch
import torch.nn as nn
import numpy as np
import time
import random
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientFedAPP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.class_sample_num = {i: 0 for i in range(args.num_classes)}
        trainloader = self.load_train_data()
        for i, (x, y) in enumerate(trainloader):
            for i, yy in enumerate(y):
                y_c = yy.item()
                self.class_sample_num[y_c] += 1
        # print(1)


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = load_item(self.role, 'personalized_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        protos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        delay_time = random.randint(self.delay_group*self.args.delay_time, (self.delay_group+1)*self.args.delay_time)
        print(delay_time)
        print(time.time() - start_time)
        self.cur_time += (delay_time + time.time() - start_time)
        self.cur_update += 1

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # local_protos = load_item(self.role, 'protos', self.save_folder_name)
        # local_protos = load_item(self.role, 'personalized_protos', self.save_folder_name)
        model.eval()
        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                pre = model.head(rep)
                predicted_classes = torch.argmax(pre, dim=1)

                #
                test_acc += ((predicted_classes == y).sum().item())

                test_num += y.shape[0]
        return test_acc, test_num, 0


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


def agg_func(protos):
    """
    Returns the average of the weights.
    """
    res = defaultdict(list)
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
            res[label] = [proto / len(proto_list), len(proto_list)]
        else:
            protos[label] = proto_list[0]
            res[label] = [proto_list[0], 1]

    return res