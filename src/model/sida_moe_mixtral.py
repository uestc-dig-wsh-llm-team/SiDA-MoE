import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.models.mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import(
    MixtralBLockSparseTop2MLP,
)
    
class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor,hash_table) -> torch.Tensor:
        # Hash table List[Tuple[Dict?,int,str,int]]  
        # 1# deamon.gen_hash[0] 
        # 2# batch_idx 
        # 3# original hash table's key 
        # 4# topk
        """ """
        hash_table, batch_idx, key, topk = hash_table.pop()
        is_soft_target = False
        local_batch_size = hidden_states.size(0)
        
        if isinstance(hash_table[key], tuple):
            is_soft_target = True
            # The hash table contains expert indices and probabilities
            selected_experts = hash_table[key][1][
                batch_idx * local_batch_size : (batch_idx + 1) * local_batch_size, :, :
            ]   # 当前Batch的index [seq_len, 专家数量]
            expert_prob = hash_table[key][0][
                batch_idx * local_batch_size : (batch_idx + 1) * local_batch_size, :, :
            ]
        else:
            selected_experts = hash_table[key][
                batch_idx * local_batch_size : (batch_idx + 1) * local_batch_size, :, :
            ]

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = torch.gather(original_routing_weights, -1, selected_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

