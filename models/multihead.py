import torch
from torch import nn
import fairseq

class Multihead_Wav2vec(nn.Module):

    def __init__(self,
    wav2vec_path ="/home/filip/speech_metrics_eval/externals/metrics/Noresqa/models/wav2vec_small.pt",
    rnn_layers=2,
    rnn_hidden=256,
    rnn_bidirectional=True,
    linear_size = 149

     ):    
        super().__init__()
        INPUT_LAYER_OUT_DIM=768
        input_layer, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        input_layer = input_layer[0]
        input_layer.remove_pretraining_modules()
        input_layer.requires_grad = False
        self.input_layer = input_layer
        self.rnn = nn.LSTM(INPUT_LAYER_OUT_DIM, rnn_hidden, rnn_layers,bidirectional=rnn_bidirectional)
        
        self.embed_dim = rnn_hidden * 2

        self.att_MOS = nn.MultiheadAttention(self.embed_dim, num_heads = 1) #Inputs: query: (L, N, E), Output: (L,N,E)
        self.att_PESQ = nn.MultiheadAttention(self.embed_dim, num_heads = 1)
        self.att_SDR = nn.MultiheadAttention(self.embed_dim, num_heads = 1)

        # Output Layer Non-intrusive Metric
        self.dense_MOS_1   = nn.Linear(linear_size * self.embed_dim, 100)
        self.dense_PESQ_1  = nn.Linear(linear_size * self.embed_dim, 100)
        self.dense_SDR_1   = nn.Linear(linear_size * self.embed_dim, 100)

        self.dense_MOS_2   = nn.Linear(100, 1) # MOS score
        self.dense_PESQ_2  = nn.Linear(100, 1) # PESQ score
        self.dense_SDR_2   = nn.Linear(100, 1) # SDR score

        self.relu = nn.ReLU()


    def forward(self, wav):
        with torch.no_grad():
            res = self.input_layer(wav, mask=False, features_only=True)
            x = res['x']

        metric_rnn_feat, _ = self.rnn(x)

        # Attention (self attention)
        att_mos, mos_att_weights = self.att_MOS(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)
        att_pesq, pesq_att_weights = self.att_PESQ(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)
        att_sdr, sdr_att_weights = self.att_SDR(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)

        att_mos   = att_mos.view(att_mos.size(0),-1)
        att_pesq  = att_pesq.view(att_pesq.size(0),-1)
        att_sdr   = att_sdr.view(att_sdr.size(0),-1) #[12,576]

        # Dense (2 layer)
        fc_mos   = self.relu(self.dense_MOS_1(att_mos)) # B 
        fc_pesq  = self.relu(self.dense_PESQ_1(att_pesq)) # B 
        fc_sdr   = self.relu(self.dense_SDR_1(att_sdr)) # B
        

        # Quality Estimation
        est_MOS   = self.relu(self.dense_MOS_2(fc_mos)) # B 
        est_PESQ  = self.relu(self.dense_PESQ_2(fc_pesq)) # B 
        est_SDR   = self.dense_SDR_2(fc_sdr) # B
        
        return est_MOS, est_PESQ, est_SDR, mos_att_weights, pesq_att_weights, sdr_att_weights
        