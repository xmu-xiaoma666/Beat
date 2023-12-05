from option.options import options, config
from data.dataloader_bert import get_dataloader
import torch
from model.model_BEAT import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import CRLoss
from torch import optim
import logging
import os
import sys
from src.test_BEAT import test_multi
from torch.autograd import Variable


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def seed_everything(seed=42):
    '''
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True



def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def train(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    config(opt)
    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'

    id_loss_fun_global = Id_Loss(opt, 1, opt.feature_length).to(opt.device)
    id_loss_fun_local = Id_Loss(opt, opt.part, opt.feature_length).to(opt.device)
    id_loss_fun_non_local = Id_Loss(opt, opt.part, 512).to(opt.device)
    cr_loss_fun = CRLoss(opt)
    network = TextImgPersonReidNet(opt).to(opt.device)

    cnn_params = list(map(id, network.ImageExtract.parameters()))
    bert_params = list(map(id, network.TextExtract.language_model.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params , network.parameters())
    other_params = list(other_params)
    other_params.extend(list(id_loss_fun_global.parameters()))
    other_params.extend(list(id_loss_fun_local.parameters()))
    other_params.extend(list(id_loss_fun_non_local.parameters()))
    param_groups = [{'params': other_params, 'lr': opt.lr},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1},
                    ]

    for p in network.TextExtract.language_model.parameters():
        p.requires_grad = False


    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_history = 0

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    for epoch in range(opt.epoch):
        # network.eval()
        # test_best = test_multi(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best)
        # network.train()
        id_loss_sum = 0
        ranking_loss_sum = 0

        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))
        
        for times, [image, label, caption, caption_code, caption_length, caption_cr, caption_code_cr, caption_length_cr] in enumerate(train_dataloader):
            """_summary_
            Args:
                image (bs, 3, 384, 128)
                label (bs, 1)
                caption_code (bs, 100)
                caption_length (bs)
                caption_code_cr (bs, 100)
                caption_length_cr (bs)
            """
            
            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))

            tokens, segments, input_masks, caption_length = network.TextExtract.language_model.pre_process(caption)
            tokens_cr, segments_cr, input_masks_cr, caption_length_cr = network.TextExtract.language_model.pre_process(caption_cr)
            tokens, segments, input_masks, caption_length = tokens.to(opt.device), segments.to(opt.device), input_masks.to(opt.device), caption_length.to(opt.device)
            tokens_cr, segments_cr, input_masks_cr, caption_length_cr = tokens_cr.to(opt.device), segments_cr.to(opt.device), input_masks_cr.to(opt.device), caption_length_cr.to(opt.device)

            """
                img_global, (bs,1024,1)
                img_local, (bs,1024,6)
                img_non_local, (bs,512,6)
                txt_global, (bs,1024,1)
                txt_local, (bs,1024,6)
                txt_non_local, (bs,512,6)
            """
            img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local, img_global_txtspace, img_local_txtspace, img_non_local_txtspace, txt_global_imgspace, txt_local_imgspace, txt_non_local_imgspace = network(image, tokens, segments, input_masks,
                                                                                                 caption_length)

            txt_global_cr, txt_local_cr, txt_non_local_cr, txt_global_cr_imgspace, txt_local_cr_imgspace, txt_non_local_cr_imgspace = network.txt_embedding(tokens_cr, segments_cr, input_masks_cr, caption_length_cr)

            id_loss_global = id_loss_fun_global(img_global, txt_global, label)
            id_loss_local = id_loss_fun_local(img_local, txt_local, label)
            id_loss_non_local = id_loss_fun_non_local(img_non_local, txt_non_local, label)
            for i in range(opt.rem_num):
                id_loss_global += id_loss_fun_global(img_global_txtspace[:,i,:,:], txt_global_imgspace[:,i,:,:], label)
                id_loss_local += id_loss_fun_local(img_local_txtspace[:,i,:,:], txt_local_imgspace[:,i,:,:], label)
                id_loss_non_local += id_loss_fun_non_local(img_non_local_txtspace[:,i,:,:], txt_non_local_imgspace[:,i,:,:], label)
            id_loss = id_loss_global + (id_loss_local + id_loss_non_local) * 0.5
            
            cr_loss_global1 = 0
            cr_loss_global2 = 0
            for i in range(opt.rem_num):
                cr_loss_global1 += cr_loss_fun(img_global, txt_global_imgspace[:,i,:,:], txt_global_cr_imgspace[:,i,:,:], label, epoch >= opt.epoch_begin)
                cr_loss_global2 += cr_loss_fun(img_global_txtspace[:,i,:,:], txt_global, txt_global_cr, label, epoch >= opt.epoch_begin)
            cr_loss_global = (cr_loss_global1 + cr_loss_global2) /2

            cr_loss_local1 = 0
            cr_loss_local2 = 0
            for i in range(opt.rem_num):
                cr_loss_local1 += cr_loss_fun(img_local, txt_local_imgspace[:,i,:,:], txt_local_cr_imgspace[:,i,:,:], label, epoch >= opt.epoch_begin)
                cr_loss_local2 += cr_loss_fun(img_local_txtspace[:,i,:,:], txt_local, txt_local_cr, label, epoch >= opt.epoch_begin)
            cr_loss_local = (cr_loss_local1 + cr_loss_local2) / 2

            cr_loss_non_local1 = 0
            cr_loss_non_local2 = 0
            for i in range(opt.rem_num):
                cr_loss_non_local1 += cr_loss_fun(img_non_local, txt_non_local_imgspace[:,i,:,:],txt_non_local_cr_imgspace[:,i,:,:], label, epoch >= opt.epoch_begin)
                cr_loss_non_local2 += cr_loss_fun(img_non_local_txtspace[:,i,:,:], txt_non_local,txt_non_local_cr, label, epoch >= opt.epoch_begin)
            cr_loss_non_local = (cr_loss_non_local1 + cr_loss_non_local2) / 2

            ranking_loss = cr_loss_global + (cr_loss_local + cr_loss_non_local) * 0.5

            optimizer.zero_grad()
            loss = (id_loss + ranking_loss)
            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Setp: %d, ranking_loss: %.2f, id_loss: %.2f"
                             % (epoch + 1, opt.epoch, times + 1, ranking_loss, id_loss))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)

        logging.info("Epoch: %d/%d , ranking_loss: %.2f, id_loss: %.2f"
                     % (epoch + 1, opt.epoch, ranking_loss_avg, id_loss_avg))

        print(opt.model_name)
        with torch.no_grad():
            network.eval()
            test_best = test_multi(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best)
            network.train()

        if test_best > test_history:
            test_history = test_best
            state = {
                'network': network.cpu().state_dict(),
                'test_best': test_best,
                'epoch': epoch,
                'WN': id_loss_fun_non_local.cpu().state_dict(),
                'WL': id_loss_fun_local.cpu().state_dict(),
            }
            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun_non_local.to(opt.device)
            id_loss_fun_local.to(opt.device)

        scheduler.step()

    logging.info('Training Done')


if __name__ == '__main__':
    seed_everything(42)
    opt = options().opt
    train(opt)

