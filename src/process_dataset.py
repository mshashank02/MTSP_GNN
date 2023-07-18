from typing import List, Tuple, Union
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset,Data,HeteroData
import numpy as np
import scipy.sparse as sp
import os


print(f"Torch version: {torch.__version__}")
print(f"Torch version: {torch_geometric.__version__}")


class MTSPDataset(Dataset):
    def __init__(self,root,transform = None, pre_transform = None):
        super(MTSPDataset, self).__init__(root,transform,pre_transform)

    @property

    def raw_file_names(self):

        return 'drones.csv''payloads.csv'
    @property

    def processed_file_names(self):

        return 'not_done.csv'
    
    def download(self):
        pass
    def process(self):
        self.data_drones = pd.read_csv(self.raw_paths[0])
        self.data_payloads = pd.read_csv(self.raw_paths[1])
        self.data = HeteroData()

        for index in range(self.data_drones.shape[0]):

            #Get drone node Features
            drone_node_features = self._get_drone_node_features()
            #Get payload node Features
            payload_node_features = self._get_payload_node_features()

            #Get Adjacency Info for drone_to_drone_edge
            drone_to_drone_edge_index = self._get_drone_to_drone_adjacency_info()
            #Get Adjacency Info for drone_to_payload_edge
            drone_to_payload_edge_index = self._get_drone_to_payload_adjacency_info()
            #Get Adjacency Info for payload_to_payload_edge
            payload_to_payload_edge_index = self._get_payload_to_payload_adjacency_info()
            

            self.data['drones'].x = drone_node_features
            self.data['payloads'].x = payload_node_features

            self.data['drones','drone_to_drone_edge','drones'] = drone_to_drone_edge_index
            self.data['drones','drone_to_payload_edge','payloads'] = drone_to_payload_edge_index
            self.data['payloads','payloads_to_payloads_edge','payloads'] = payload_to_payload_edge_index
            
            

            torch.save(self.data,
                       os.path.join(self.processed_dir,
                                    f'data_{index}.pt'))
         
    def _get_drone_node_features(self):
        
        all_drone_node_feats = []

        for i in range(self.data_drones.shape[0]):

            drone_node_feats = []

            #Drone capacity
            drone_node_feats.append(self.data_drones['drone_position'][i])

            #Drone_depot_position
            drone_node_feats.append(self.data_drones['drone_depot_position'][i])

            #Drone_avg_velocity
            drone_node_feats.append(self.data_drones['drone_avg_veocity'][i])

            all_drone_node_feats.append(drone_node_feats)
            
        all_drone_node_feats = np.asarray(all_drone_node_feats)
        return torch.tensor(all_drone_node_feats, dtype=torch.float)

    def _get_payload_node_features(self):
        
        all_payload_node_feats = []

        for i in range(self.data_payloads.shape[0]):

            payload_node_feats = []
            
            #Payload_weight
            payload_node_feats.append(self.data_payloads['payload_weight'][i])

            #Payload_source
            payload_node_feats.append(self.data_payloads['payload_source'][i])

            #Payload_Destination
            payload_node_feats.append(self.data_payloads['payload_destination'][i])

            all_payload_node_feats.append(payload_node_feats)
        
        all_payload_node_feats = np.asarray(all_payload_node_feats)
        return torch.tensor(all_payload_node_feats,dtype = torch.float)
    
    def _get_drone_to_drone_adjacency_info(self):

        self.number_drone_nodes = self.data['drones'].num_nodes 
        # Calculate the number of edges (including self-loops)
        num_edges = self.number_drone_nodes * self.number_drone_nodes 

        # Create arrays to store the source and target node indices
        source_indices = torch.zeros(num_edges, dtype=torch.long)
        target_indices = torch.zeros(num_edges, dtype=torch.long)

        # Fill the arrays with the source and target indices for all edges
        idx = 0
        for i in range(self.number_drone_nodes):
            for j in range(self.number_drone_nodes):
                source_indices[idx] = i
                target_indices[idx] = j
                idx += 1

        # Create the edge index tensor
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        return edge_index
    
    def _get_drone_to_payload_adjacency_info(self):
        
        self.number_payload_nodes = self.data['payloads'].num_nodes
        type_of_edge = "payload_to drone_bidirectional"


        if type_of_edge == "payload_to_drone_bidirectional":

            
            num_edges = 2*self.number_drone_nodes * self.number_payload_nodes 

            # Create arrays to store the source and target node indices
            source_indices = torch.zeros(num_edges, dtype=torch.long)
            target_indices = torch.zeros(num_edges, dtype=torch.long)

            # Fill the arrays with the source and target indices for all edges
            idx = 0
            for i in range(self.number_drone_nodes):
                for j in range(self.number_payload_nodes):
                    source_indices[idx] = i
                    target_indices[idx] = j + self.number_drone_nodes
                    idx += 1
                    source_indices[idx] = j + self.number_drone_nodes # Bidirectional edge
                    target_indices[idx] = i
                    idx += 1

            # Create the edge index tensor
            edge_index = torch.stack([source_indices, target_indices], dim=0)
        
        elif type_of_edge == "payload_to drone_only":
            num_edges = self.number_drone_nodes*self.number_payload_nodes

            source_indices = torch.zeros(num_edges, dtype = torch.long)
            target_indices = torch.zeros(num_edges, dtype = torch.long)

            idx = 0
            for i in range(self.number_drone_nodes):
                for j in range(self.number_payload_nodes):
                    source_indices[idx] = j + self.number_drone_nodes
                    target_indices[idx] = i
                    idx += 1
            # Create the edge index tensor
            edge_index = torch.stack([source_indices, target_indices], dim=0)
        
        elif type_of_edge == "drone_to_payload_only":
            num_edges = self.number_drone_nodes*self.number_payload_nodes

            source_indices = torch.zeros(num_edges, dtype = torch.long)
            target_indices = torch.zeros(num_edges, dtype = torch.long)

            idx = 0
            for i in range(self.number_drone_nodes):
                for j in range(self.number_payload_nodes):
                    source_indices[idx] = i
                    target_indices[idx] = j + self.number_drone_nodes
                    idx += 1
            # Create the edge index tensor
            edge_index = torch.stack([source_indices, target_indices], dim=0)
                    


        return edge_index
    
    def _get_payload_to_payload_adjacency_info(self):

        type_of_edge = "bidirectional"


        if type_of_edge == "bidirectional":

            # Calculate the number of edges (including self-loops)
            num_edges = self.number_payload_nodes * self.number_payload_nodes 

            # Create arrays to store the source and target node indices
            source_indices = torch.zeros(num_edges, dtype=torch.long)
            target_indices = torch.zeros(num_edges, dtype=torch.long)

            # Fill the arrays with the source and target indices for all edges
            idx = 0
            for i in range(self.number_payload_nodes):
                for j in range(self.number_payload_nodes):
                    source_indices[idx] = i
                    target_indices[idx] = j
                    idx += 1

            # Create the edge index tensor
            edge_index = torch.stack([source_indices, target_indices], dim=0)
        
        else:
            edge_index = torch.tensor([],
                                      [],
                                      dtype= torch.long)

        return edge_index
    
