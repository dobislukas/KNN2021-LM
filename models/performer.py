import torch

def create_projection_matrix(m, d, scaling=0):
    """
    m: number of random projections
    d: dimensionality of each projection
    """
    nb_full_blocks = int(m / d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        q = q.t()
        block_list.append(q)

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        q = q.t()
        block_list.append(q[0:remaining_rows])

    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(float(d)) * torch.ones((m))
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)
    return torch.diag(multiplier) @ final_matrix

def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.
    Computes random features for the softmax kernel using FAVOR+ mechanism from
    https://arxiv.org/pdf/2009.14794.pdf.
    Args:
        data: input data tensor of the shape [B, L, H, D], where: B - batch
        dimension, L - attention dimensions, H - heads, D - features.
        is_query: indicates whether input data is a query oor key tensor.
        projection_matrix: random Gaussian matrix of shape [M, D], where M stands
        for the number of random features and each D x D sub-block has pairwise
        orthogonal rows.
        numerical_stabilizer: small positive constant for numerical stability.
    Returns:
        Corresponding kernel feature map.
    """
    data_normalizer = data.shape[-1]**-0.25
    data = data_normalizer * data
    ratio = projection_matrix.shape[0]**-0.5
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(
                data_dash, dim=-1, keepdims=True).values ) + numerical_stabilizer)
    else:
        max_vals = torch.max(torch.max(data_dash, dim=1, keepdims=True).values, dim=3, keepdims=True).values
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - max_vals) +
            numerical_stabilizer)

    return data_dash

def causal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR causal attention A_{masked}V.
    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].
        vs: value tensor of the shape [L,B,H,D].
    Returns:
        Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    for index in range(qs.shape[0]):
        sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index]).unsqueeze(0))

    result = torch.cat(result, dim=0)
    return result

def causal_denominator(qs, ks):
    """Computes FAVOR normalizer in causal attention.
    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].
    Returns:
        FAVOR normalizer in causal attention.
    """

    result = []
    sums = torch.zeros_like(ks[0])

    for index in range(qs.shape[0]):
        sums = sums + ks[index]
        result.append( torch.sum(qs[index] * sums, dim=2).unsqueeze(0) )

    result = torch.cat(result, axis=0)
    return result

def favor_attention(query,
                    key,
                    value,
                    projection_matrix=None):
    """Computes FAVOR normalized attention.
    Args:
        query: query tensor.
        key: key tensor.
        value: value tensor.
        projection_matrix: projection matrix to be used.
    Returns:
        FAVOR normalized attention.
    """
    query_prime = softmax_kernel_transformation(query, True,
                                        projection_matrix)  # [B,L,H,M]
    key_prime = softmax_kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
    query_prime = torch.transpose(query_prime, 1, 0)  # [L,B,H,M]
    key_prime = torch.transpose(key_prime, 1, 0)  # [L,B,H,M]
    value = torch.transpose(value, 1, 0)  # [L,B,H,D]


    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)

    av_attention = torch.transpose(av_attention, 1, 0)
    attention_normalizer = torch.transpose(attention_normalizer, 1, 0)
    attention_normalizer = attention_normalizer.unsqueeze(-1)
    return av_attention / attention_normalizer, None

class FavorSelfAttention(torch.nn.Module):
    def __init__(self, d_k, nb_random_features):
        super().__init__()
        self.d_k = d_k
        self.nb_random_features = nb_random_features
        projection_matrix = create_projection_matrix(
            self.nb_random_features, self.d_k)
        self.register_buffer('projection_matrix', projection_matrix)
    @torch.no_grad()
    def redraw(self):
        projection_matrix = create_projection_matrix(
            self.nb_random_features, self.d_k)
        self.projection_matrix.copy_(projection_matrix)
        del projection_matrix
    def forward(self, q, k, v):
        # |q| : (batch_size, n_heads, q_len, d_k)
        # |k| : (batch_size, n_heads, k_len, d_k)
        # |v| : (batch_size, n_heads, v_len, d_v)
        return favor_attention(q, k, v, self.projection_matrix)
        

class MultiHeadAttentionPerformer(torch.nn.Module):
    def __init__(self, d_model, n_heads, attn_pdrop, nb_random_features=256):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = torch.nn.Linear(d_model, d_model)
        self.WK = torch.nn.Linear(d_model, d_model)
        self.WV = torch.nn.Linear(d_model, d_model)
        self.favor_attention = FavorSelfAttention(self.d_k, nb_random_features)
        self.linear = torch.nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, att_mask):
        # |Q| : (batch_size, q_len(=seq_len), d_model)
        # |K| : (batch_size, k_len(=seq_len), d_model)
        # |V| : (batch_size, v_len(=seq_len), d_model)
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v)
        # |q_heads| : (batch_size, q_len, n_heads, d_k), |k_heads| : (batch_size, q_len, n_heads, d_k), |v_heads| : (batch_size, q_len, n_heads, d_k)
        
        attn, att_weights = self.favor_attention(q_heads, k_heads, v_heads)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        attn = attn.view(batch_size, -1, self.n_heads * self.d_v)
        # |attn| : (batch_size, q_len, n_heads * d_v = d_model)
        outputs = self.linear(attn)
        # |outputs| : (batch_size, q_len, d_model)
        return outputs, att_weights
