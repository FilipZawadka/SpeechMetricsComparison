import sys
import os
import tensorboard

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.multihead import Multihead_Wav2vec
from dataset.dataset import NISQA_Corpus_Dataset

from tensorboardX import SummaryWriter


def training_loop():

    target_directory = '/home/filip/speech_metrics_eval/training_checkpoints/multihead_wav2vec_11'
    os.makedirs(target_directory)

    model = Multihead_Wav2vec()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    mse = torch.nn.MSELoss()

    train_dataset = NISQA_Corpus_Dataset(clip_sec=3)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=3, collate_fn=None)

    val_dataset = NISQA_Corpus_Dataset(csv_path="/work/data/speech_metrics_eval/NISQA_Corpus/NISQA_VAL_SIM/NISQA_VAL_SIM_file_pesq_si_sdr.csv",
                                        clip_sec=3)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, sampler=None,
        batch_sampler=None, num_workers=5, collate_fn=None)

    mos_loss_weight = 10
    pesq_loss_weight = 1
    sdr_loss_weight = 0.1

    writer = SummaryWriter(logdir=target_directory)

    for step, (audio, mos, pesq_val, si_sdr_val) in enumerate(train_dataloader):

        est_MOS, est_PESQ, est_SDR, mos_att_weights, pesq_att_weights, sdr_att_weights = model(audio)

        mos_loss = mse(mos.to(torch.float32), est_MOS.to(torch.float32))
        pesq_loss = mse(pesq_val.to(torch.float32),est_PESQ.to(torch.float32))
        sdr_loss = mse(si_sdr_val.to(torch.float32),est_SDR.to(torch.float32))

        loss = mos_loss * mos_loss_weight + pesq_loss * pesq_loss_weight + sdr_loss * sdr_loss_weight
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        
        optimizer.step()

        writer.add_scalar('loss', loss, step)

        writer.add_scalar('mos_loss', mos_loss, step)
        writer.add_scalar('pesq_loss', pesq_loss, step)
        writer.add_scalar('sdr_loss', sdr_loss, step)

        if step % 100 == 0:
            print(step,"\n")

            with torch.no_grad():
                audio, mos, pesq_val, si_sdr_val = next(iter(val_dataloader))
                est_MOS, est_PESQ, est_SDR, mos_att_weights, pesq_att_weights, sdr_att_weights = model(audio)

                mos_loss = mse(mos.to(torch.float32), est_MOS.to(torch.float32))
                pesq_loss = mse(pesq_val.to(torch.float32),est_PESQ.to(torch.float32))
                sdr_loss = mse(si_sdr_val.to(torch.float32),est_SDR.to(torch.float32))

                loss = mos_loss * mos_loss_weight + pesq_loss * pesq_loss_weight + sdr_loss * sdr_loss_weight

                writer.add_scalar('validation_loss', loss, step)

                writer.add_scalar('validation_mos_loss', mos_loss, step)
                writer.add_scalar('validation_pesq_loss', pesq_loss, step)
                writer.add_scalar('validation_sdr_loss', sdr_loss, step)

            print("loss: ", loss, "\n")


        if step % 500 == 0:
            torch.save({
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 
                os.path.join(target_directory, f"checkpoint_{step}.pt"))


        



if __name__ == "__main__":
    training_loop()