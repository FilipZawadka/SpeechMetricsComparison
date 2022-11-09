from torch import nn
import fairseq

class Multihead_Wav2vec(nn.Module):

    def __init__(self,
    wav2vec_path ="/home/filip/speech_metrics_eval/externals/metrics/Noresqa/models/wav2vec_small.pt",
    rnn_layers=2,
    rnn_hidden=256,
    rnn_bidirectional=True
     ):    
        super().__init__()
        INPUT_LAYER_OUT_DIM=768
        input_layer, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        input_layer = input_layer[0]
        input_layer.remove_pretraining_modules()
        self.input_layer = input_layer
        self.rnn = nn.LSTM(INPUT_LAYER_OUT_DIM, rnn_hidden, rnn_layers,bidirectional=rnn_bidirectional)




    def forward(self, wav):
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']