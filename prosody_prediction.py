import torch
import numpy as np
from torch import nn
from torch.optim import SGD
from models import Poly_Phoneme_Classifier
from sklearn.preprocessing import StandardScale
from sklearn.model_selection import train_test_split 

from pytorch_pretrained_bert import BertModel



# NN
# 搭建韵律预测网络
# 模型不传入超参，这一版超参写死在类内
# in: 字编号序列, (none,)
# out: 每个字属于韵律类别后验概率, (none, 5)
class Prosody(nn.Module):
    def __init__(self):
        super (prosody,self).__init__()
        ## 定义bert层
        # in (none, )   
        # out (none, 512)
        # word -> embedding, use Pytorch Bert
        self.bert = BertModel.from_pretrained('./bert/bert-base-chinese')


        ## 定义transformer层
        # (none, 512) -> (none, 1024 + 300) -> (none, 1024) -> (none, 106)
        # feature extract, use Poly_Phoneme_Classifier as transformer
        # 注意：不需要 n_pinyin_symbols, 只是传进去个 1, 其实无意义
        # 注意： Transformer 超参数在类内定死，只能在外加 Linear
        class TMP_HPARAMS(object):
            def __init__(self):
                self.n_pinyin_symbols = 1
        tmp_hparams = TMP_HPARAMS()
        self.transformer_linear_pre = nn.Linear(in_features=512, out_features=1024 + 300)
        self.transformer = Poly_Phoneme_Classifier(hparams=tmp_hparams)
        self.transformer_linear_post = nn.Linear(in_features = 1024, out_features = 106)


        ## 定义线性层
        # (none, 106) -> (none, 5)
        self.linear = nn.Linear(in_features = 106, out_features = 5)
        

        ## 定义Gumbel_Softmax层/Softmax层
        # 最后一维归一化
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, x):
        # bert
        # [0][0] 是取出 hidden 的特殊用法
        # (none,) -> (none, 512)
        x = self.bert(x)[0][0]


        # transformer
        # 复用的 Poly_Phoneme_Classifier， 因此前后加了 Linear。 TODO
        # (none, 512) -> (none, 1024 + 300) -> (none, 1024) -> (none, 106)
        x = self.transformer_linear_pre(x)
        x = self.transformer.forward_train_polyphonic(x)
        x = self.transformer_linear_post(x)


        # linear + softmax
        # 变为 5 类后验概率, #0, #1, #2, #3, #4
        # (none, 106) -> (none, 5)
        logist = self.linear(x)
        p = self.softmax(logist)
        return p




def main():
    return
    ## 导入数据
    prosody_data = 0

    ## 数据切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        prosody_data.data, prosody_data.target, test_size = 0.3, random_state = 42)

    ## 数据标准化处理
    scale = StandardScale()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.fit_transform(X_test)

    ## 将数据集转化为张量
    train_xt = torch.from_numpy(X_train_s.astype(np.float32))
    train_yt = torch.from_numpy(Y_train.astype(np.float32))
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))

    ## 将训练数据处理为数据加载器
    train_data = Data.TensorDataset(train_xt,train_yt)
    test_data = Data.TensorDataset(test_xt,test_yt)
    train_loader = Data.DataLoader(dataset = train_data, batch_size = 64, shuffle = True, num_workers = 1)







    prosody = Prosody()

    ## 定义优化器
    optimizer = torch.optim.SGD(prosody.parameters(),lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    train_loss_all = []
    ## 对模型进行迭代训练，对所有的数据训练epoch轮
    for epoch in range(30):
        train_loss = 0
        train_number = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            output = prosody(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)

    ## 对测试集进行预测
    pre_y = prosody(test_xt)

