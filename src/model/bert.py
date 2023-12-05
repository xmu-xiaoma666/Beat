import torch
from torch import nn

from pytorch_transformers import BertModel, BertConfig, BertTokenizer


class Bert(nn.Module):
    def __init__(self,opt): 
        super(Bert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('./src/pretrained/bert-base-uncased-vocab.txt')
        modelConfig = BertConfig.from_pretrained('./src/pretrained/bert_config.json')
        self.textExtractor = BertModel.from_pretrained(
            './src/pretrained/pytorch_model.bin', config=modelConfig)
        self.opt = opt
        
    def pre_process(self, texts):

        tokens, segments, input_masks, text_length = [], [], [], []
        for text in texts:
            text = '[CLS] ' + text + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) > self.opt.caption_length_max:
                indexed_tokens = indexed_tokens[:self.opt.caption_length_max]
                
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))


        for j in range(len(tokens)):
            padding = [0] * (100 - len(tokens[j]))
            text_length.append(min(self.opt.caption_length_max,len(tokens[j])+3))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        
        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        input_masks = torch.tensor(input_masks)
        text_length = torch.tensor(text_length)

        return tokens, segments, input_masks, text_length


    def forward(self, tokens, segments, input_masks):
        
        output=self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=input_masks)
        text_embeddings = output[0]

        return text_embeddings

if __name__ == '__main__':
    bert = Bert()
    tokens, segments, input_masks, text_length = bert.pre_process(["i am a people haha haha, good","i am a people haha haha, good"])
    text_embeddings = bert(tokens, segments, input_masks) #bs, 100, 768
    print(text_embeddings.shape)

