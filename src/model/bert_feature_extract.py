from torch import nn
import torch
from .bert import Bert


class TextExtract(nn.Module):

    def __init__(self, opt):
        super(TextExtract, self).__init__()

        # self.embedding_local = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        # self.embedding_global = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.language_model = Bert(opt)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(768, 2048, num_layers=1, bidirectional=True, bias=False)

    def forward(self, tokens, segments, input_masks, text_length):

        text_embedding = self.language_model(tokens, segments, input_masks) #bs, 100, 768
        text_embedding_local = self.dropout(text_embedding) #bs, 100, 768
        text_embedding_local = self.calculate_different_length_lstm(text_embedding_local, text_length, self.lstm)#bs, 2048, 100, 1
        text_embedding_global = text_embedding_local[:,:,0,:].unsqueeze(2)  #bs, 2048, 1, 1

        return text_embedding_global, text_embedding_local

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):
        text_length = text_length.view(-1) # 64
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :] #bs, 100, 512
        sort_text_length = text_length[sort_index] #bs
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length.cpu(),
                                                                  batch_first=True)

        
        # self.lstm.flatten_parameters()
        packed_feature, _ = lstm(packed_text_embedding)  # [hn, cn] #1479,4096
        total_length = text_embedding.size(1)
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature,
                                                        batch_first=True,
                                                        total_length=total_length)  # including[feature, length]

        unsort_feature = sort_feature[0][unsort_index, :] #bs, 100, 4096
        unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                          + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2 #bs, 100, 2048

        return unsort_feature.permute(0, 2, 1).contiguous().unsqueeze(3) #bs. 2048, 100, 1




