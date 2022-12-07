import sys
import os
import numpy as np
import random

import tensorboard

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.multihead import Multihead_Wav2vec
from dataset.dataset import NISQA_Corpus_Dataset

from tensorboardX import SummaryWriter

random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

def training_loop():

    target_directory = '/work/checkpoints/speech_evaluation_metrics/training_checkpoints/multihead_wav2vec_14'
    os.makedirs(target_directory)

    model = Multihead_Wav2vec()

    # chckpt = torch.load("/home/filip/speech_metrics_eval/training_checkpoints/multihead_wav2vec_7/checkpoint_1000.pt")    
    # model.load_state_dict(chckpt["model_state_dict"])
    # model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse = torch.nn.MSELoss()

    train_dataset = NISQA_Corpus_Dataset(clip_sec=5)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=4, collate_fn=None)

    val_dataset = NISQA_Corpus_Dataset(csv_path="/work/data/speech_metrics_eval/NISQA_Corpus/NISQA_VAL_SIM/NISQA_VAL_SIM_file_pesq_si_sdr.csv",
                                        clip_sec=5)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, sampler=None,
        batch_sampler=None, num_workers=2, collate_fn=None)

    mos_loss_weight = 10
    pesq_loss_weight = 1
    sdr_loss_weight = 0.1

    writer = SummaryWriter(logdir=target_directory)
    max_epochs = 1000
    glob_step = 0

    for epoch in range(max_epochs):
        print("Epoch: ",epoch)
        for step, (audio, mos, pesq_val, si_sdr_val) in enumerate(train_dataloader):
            glob_step += 1
            est_MOS, est_PESQ, est_SDR, mos_att_weights, pesq_att_weights, sdr_att_weights = model(audio)

            mos_loss = mse(mos.to(torch.float32), est_MOS.to(torch.float32))
            pesq_loss = mse(pesq_val.to(torch.float32),est_PESQ.to(torch.float32))
            sdr_loss = mse(si_sdr_val.to(torch.float32),est_SDR.to(torch.float32))

            loss = mos_loss * mos_loss_weight + pesq_loss * pesq_loss_weight + sdr_loss * sdr_loss_weight
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            
            optimizer.step()
            writer.add_scalar('loss', loss, glob_step)

            writer.add_scalar('mos_loss', mos_loss, glob_step)
            writer.add_scalar('pesq_loss', pesq_loss, glob_step)
            writer.add_scalar('sdr_loss', sdr_loss, glob_step)

            if step % 100 == 0:
                print(glob_step,"\n")

                with torch.no_grad():
                    audio, mos, pesq_val, si_sdr_val = next(iter(val_dataloader))
                    #TODO: preview att
                    est_MOS, est_PESQ, est_SDR, mos_att_weights, pesq_att_weights, sdr_att_weights = model(audio)

                    mos_loss = mse(mos.to(torch.float32), est_MOS.to(torch.float32))
                    pesq_loss = mse(pesq_val.to(torch.float32),est_PESQ.to(torch.float32))
                    sdr_loss = mse(si_sdr_val.to(torch.float32),est_SDR.to(torch.float32))

                    loss = mos_loss * mos_loss_weight + pesq_loss * pesq_loss_weight + sdr_loss * sdr_loss_weight

                    writer.add_scalar('validation_loss', loss, glob_step)

                    writer.add_scalar('validation_mos_loss', mos_loss, glob_step)
                    writer.add_scalar('validation_pesq_loss', pesq_loss, glob_step)
                    writer.add_scalar('validation_sdr_loss', sdr_loss, glob_step)

                print("loss: ", loss, "\n")


        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 
                os.path.join(target_directory, f"checkpoint_{glob_step}.pt"))


        



if __name__ == "__main__":
    training_loop()
