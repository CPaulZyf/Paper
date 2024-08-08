import time
import numpy as np
from flcore.clients.clientfedapp import clientFedAPP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import heapq
import torch
import torch.nn.functional as F


class FedAPP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_clients(clientFedAPP)

        self.class_sample_num_all = {i: 0 for i in range(args.num_classes)}
        for client in self.clients:
            for key in self.class_sample_num_all:
                self.class_sample_num_all[key] += client.class_sample_num[key]

        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

    def train(self):

        self.selected_clients = self.clients
        cur_time = time.time()
        for client in self.selected_clients:
            self.client_queue.put((time.time() - cur_time, client))
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()
                if not self.check_stop():
                    print('-' * 40 + 'Early Stop' + '-' * 40)
                    break

            self.selected_clients = self.select_clients_deay()
            print(f"\n-------------Participating Client: -------------")
            print([client.id for client in self.selected_clients])

            for client in self.selected_clients:
                client.train()
                self.client_queue.put((client.cur_time, client))

            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        self.print_result()

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)

            uploaded_protos.append(protos)

        global_protos_old = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = self.proto_aggregation(uploaded_protos, global_protos_old)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
        self.personalized_aggregation(global_protos)

    def personalized_aggregation(self, global_protos):
        for client in self.selected_clients:
            personalized_proto = defaultdict(list)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for key, (proto, num_sample) in protos.items():
                assert num_sample > 0
                assert self.class_sample_num_all[key] > 0
                beta = num_sample / self.class_sample_num_all[key]
                personalized_proto[key] = beta * proto + (1 - beta) * global_protos[key]
            save_item(personalized_proto, client.role, 'personalized_protos', self.save_folder_name)

    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
    def proto_aggregation(self, local_protos_list, global_protos_old):
        proto_coolect = defaultdict(list)
        proto_coolect_num = defaultdict(list)
        for (i, local_protos) in enumerate(local_protos_list):
            staleness = self.selected_clients[i].cur_time / \
                        self.selected_clients[i].cur_update
            for label in local_protos.keys():
                proto_coolect[label].append((local_protos[label][0], staleness))
                proto_coolect_num[label].append(local_protos[label][1])

        agg_protos_label = defaultdict(list)

        for [label, proto_list_] in proto_coolect.items():

            if len(proto_list_) > 1:
                # prototype staleness
                weights_staleness = []
                proto_list = []
                for (proto, w) in proto_list_:
                    weights_staleness.append(w)
                    proto_list.append(proto)
                total_weights_staleness = sum(weights_staleness)
                weights_staleness = [x / total_weights_staleness for x in weights_staleness]
                # prototype quality
                if global_protos_old is not None and label in global_protos_old:
                    weights_quality = calculate_similarity(global_protos_old[label], proto_list)
                else:
                    weights_quality = [1 for i in range(len(proto_list))]

                over_weights = [x * y for x, y in zip(weights_quality, weights_staleness)]
                over_weights = torch.tensor(over_weights, dtype=torch.float32)
                over_weights = F.softmax(over_weights, dim=0)
                weighted_tensors = [t * w for t, w in zip(proto_list, over_weights)]
                weighted_sum = sum(weighted_tensors)
                agg_protos_label[label] = weighted_sum.data
            else:
                a, b = proto_list_[0]
                agg_protos_label[label] = a.data

        if global_protos_old is not None:
            for key in agg_protos_label:
                if key not in global_protos_old:
                    global_protos_old[key] = agg_protos_label[key]
                else:
                    # aggregation
                    num_samples = 0
                    for clients in self.selected_clients:
                        num_samples += clients.class_sample_num[key]
                    assert num_samples > 0
                    assert self.class_sample_num_all[key] > 0
                    weights = num_samples / self.class_sample_num_all[key]
                    aggregated_label = weights * agg_protos_label[key] + (1 - weights) * global_protos_old[key]
                    global_protos_old[key] = aggregated_label
        else:
            global_protos_old = agg_protos_label

        return global_protos_old


def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))


def calculate_distances(reference_tensor, tensor_list):
    distances = torch.tensor([euclidean_distance(reference_tensor, t) for t in tensor_list])
    return distances


def calculate_similarity(reference_tensor, tensor_list):
    distances = calculate_distances(reference_tensor, tensor_list)
    similarities = F.softmax(-distances, dim=0)
    return similarities
