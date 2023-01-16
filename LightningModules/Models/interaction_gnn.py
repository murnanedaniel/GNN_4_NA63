import warnings
import torch
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, knn_graph, radius_graph
import torch.nn.functional as F

from LightningModules.utils import make_mlp
from ..gnn_base import GNNBase


class InteractionGNN(GNNBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        self.setup_layer_sizes()
        self.get_output_structure()

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            self.concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_network = make_mlp(
            self.aggregation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def aggregation_step(self, num_nodes):

        aggregation_dict = {
            "sum_mean_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "sum_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_sum": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "sum": lambda x, y, **kwargs: scatter_add(x, y, dim=0, dim_size=num_nodes),
            "mean": lambda x, y, **kwargs: scatter_mean(x, y, dim=0, dim_size=num_nodes),
            "max": lambda x, y, **kwargs: scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
        }

        return aggregation_dict[self.hparams["aggregation"]]

    def message_step(self, x, start, end, e):

        # Compute new node features
        edge_messages = self.aggregation_step(x.shape[0])(e, end)
        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, batch, all_x = None):

        if all_x is not None and self.hparams["concat_all_layers"]:
            all_x = torch.cat(all_x, dim=-1)
        else:
            all_x = x

        if self.hparams["aggregation"] == "mean_sum":
            graph_level_inputs = torch.cat([global_add_pool(all_x, batch), global_mean_pool(all_x, batch), global_max_pool(all_x, batch)], dim=1)
        elif self.hparams["aggregation"] == "sum":
            graph_level_inputs = global_add_pool(all_x, batch)
        elif self.hparams["aggregation"] == "mean":
            graph_level_inputs = global_mean_pool(all_x, batch)

        # Add dropout
        if "final_dropout" in self.hparams and self.hparams["final_dropout"] > 0.0:
            graph_level_inputs = F.dropout(graph_level_inputs, p=self.hparams["final_dropout"], training=self.training)

        return self.output_network(graph_level_inputs)

    def forward(self, batch, **kwargs):

        x = batch.x[:, :self.hparams["spatial_channels"]]
        start, end = batch.edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for _ in range(self.hparams["n_graph_iters"]):

            x, e = checkpoint(self.message_step, x, start, end, e)

        return self.output_step(x, batch.batch)

    def setup_layer_sizes(self):

        if self.hparams["aggregation"] == "sum_mean_max":
            self.concatenation_factor = 4
        elif self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]:
            self.concatenation_factor = 3
        elif self.hparams["aggregation"] in ["sum", "mean", "max"]:
            self.concatenation_factor = 2
        else:
            raise ValueError("Aggregation type not recognised")

    def get_output_structure(self):
        """
        Calculate the size of the final encoded layer that needs to be decoded.
        If we don't concat all layers, then it is simply the size of the final layer.
        TODO: If we do concat all layers, then it is the sum of all layer output sizes (the second entry in each layer shape pair).
        """

        if self.hparams["aggregation"] == "mean_sum":
            self.aggregation_factor = 3
        elif self.hparams["aggregation"] in ["mean", "sum"]:
            self.aggregation_factor = 1