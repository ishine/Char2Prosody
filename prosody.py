import torch
from pytorch_pretrained_bert import BertTokenizer,BertModel
import parse_nk


def get_bert(bert_model, bert_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    print('CHECK bert_model name:', bert_model)
    if bert_model.endswith('.tar.gz'):
        tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    bert = BertModel.from_pretrained(bert_model)
    return tokenizer, bert

class prosody_predict(nn.Module):
    def __init__(self,):
        super(prosody_predict,self).__init__()
        self.

    def forward(self,):
