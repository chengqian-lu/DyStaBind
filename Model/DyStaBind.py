import argparse
import sys
import os

from Model.ConvBlock import ConvBlock
from Model.FeatureFusionBlock import FeatureFusionBlock
from Model.M2SCA import M2SCA
from Model.PCAC import PCAC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn





class DyStaBind(nn.Module):
    def __init__(self, data_dropout, mutiscale_dropout, sequenceLengthdim, args):
        super().__init__()

        # n = 0
        in_ch = 128 # mutiscale的输入通道大小
        out_ch = 128
        if args.structure:  # 二级结构标注——one-hot编码
            self.conv_str = ConvBlock(6, out_ch=in_ch, kernel_size=3, stride_size=1, padding_size=0,
                                      dropout=data_dropout)
            self.mutiscale_str = M2SCA(in_channel=in_ch, out_channel=out_ch,
                                       dropout=mutiscale_dropout,
                                       sequenceLengthdim=sequenceLengthdim)

        if args.z_curve:
            self.conv_z_curve = ConvBlock(3, out_ch=in_ch, kernel_size=3, stride_size=1, padding_size=0,
                                          dropout=data_dropout)
            self.mutiscale_zcurve = M2SCA(in_channel=in_ch, out_channel=out_ch,
                                          dropout=mutiscale_dropout,
                                          sequenceLengthdim=sequenceLengthdim)

        if args.BERTEmbedding:
            self.conv_bert = ConvBlock(768, out_ch=in_ch, kernel_size=1, stride_size=1, padding_size=0,
                                       dropout=data_dropout)
            self.mutiscale_emb = M2SCA(in_channel=in_ch, out_channel=out_ch,
                                       dropout=mutiscale_dropout,
                                       sequenceLengthdim=sequenceLengthdim)

        if args.icSHAPE:
            self.conv_icshape = ConvBlock(1, out_ch=in_ch, kernel_size=3, stride_size=1, padding_size=0,
                                          dropout=data_dropout)
            self.mutiscale_icshape = M2SCA(in_channel=in_ch, out_channel=out_ch,
                                           dropout=mutiscale_dropout,
                                           sequenceLengthdim=sequenceLengthdim)



        self.conv_dy = ConvBlock(128*6*2, 128*6, kernel_size=1, stride_size=1, padding_size=0, dropout=data_dropout)
        self.conv_st = ConvBlock(128*6*2, 128*6, kernel_size=1, stride_size=1, padding_size=0, dropout=data_dropout)
        self.feature_fusion = FeatureFusionBlock(out_channels=128 * 6)

        self.adapavgpool = nn.AdaptiveAvgPool1d(1)

        ch = 128*6
        self.dpcnn = PCAC(filter_num=ch)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, structure, z_curve, bert_embedding, icshape, device='cpu'):
        x0, x1, x7, x8 = None, None, None, None

        if len(structure[0]) > 0:
            x0 = structure  # (N, 6, 101)
            x0 = self.conv_str(x0)
            x0 = self.mutiscale_str(x0)

        if len(z_curve[0]) > 0:
            x1 = z_curve  # (N, 3, 101)
            x1 = self.conv_z_curve(x1)
            x1 = self.mutiscale_zcurve(x1)

        if len(bert_embedding[0]) > 0:
            x7 = bert_embedding # (N, 768, 101)
            x7 = self.conv_bert(x7)
            x7 = self.mutiscale_emb(x7)

        if len(icshape[0]) > 0:
            x8 = icshape # (N, 1, 101)
            x8 = self.conv_icshape(x8)
            x8 = self.mutiscale_icshape(x8)


        if x0 is not None and x1 is not None:
            x_st = torch.cat((x0, x1), dim=1)
            x_st = self.conv_st(x_st)
        elif x0 is not None:
            x_st = x0
        elif x1 is not None:
            x_st = x1
        else:
            x_st = None

        if x7 is not None and x8 is not None:
            x_dy = torch.cat((x7, x8), dim=1)
            x_dy = self.conv_dy(x_dy)
        elif x7 is not None:
            x_dy = x7
        elif x8 is not None:
            x_dy = x8
        else:
            x_dy = None

        if x_dy is not None and x_st is not None:
            x = self.feature_fusion(x_dy, x_st, device)
            logist = self.dpcnn(x)
        elif x_dy is not None:
            logist = self.dpcnn(x_dy)
        elif x_st is not None:
            logist = self.dpcnn(x_st)
        else:
            logist = 0
        return logist



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to MREUNet!')
    parser.add_argument('--data_file', default='AARS_K562', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='./results/encode_s', type=str, help='The data path')
    parser.add_argument('--model_save_path', default='./results/encode_compare_model_MCUNet', type=str,
                        help='Save the trained model for dynamic prediction')
    parser.add_argument('--output_dir', default='./results/output_MCUNet', type=str,
                        help='Save output directory')
    parser.add_argument('--BERT_model_path', default='./BERT_Model', type=str,
                        help='BERT model path, in case you have another BERT')
    parser.add_argument('--middel_results_save_path', default='./results/middel_results_MCUNet', type=str,
                        help='Middel results(AUC, ACC, Epoch) for each epoch')

    parser.add_argument('--structure', default=False, action='store_true')
    parser.add_argument('--z_curve', default=False, action='store_true')
    parser.add_argument('--icSHAPE', default=False, action='store_true')
    parser.add_argument('--BERTEmbedding', default=False, action='store_true')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    # parser.add_argument('--dynamic_validate', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--head', type=int, default=8)

    args = parser.parse_args()

    batch_size = 8
    model = DyStaBind(data_dropout=0.3, mutiscale_dropout=0.1, sequenceLengthdim=99, args=args).to('cpu')
    x0 = torch.randn(batch_size, 4, 101)
    x1 = torch.randn(batch_size, 3, 101)
    x7 = torch.randn(batch_size, 768, 99)
    x8 = torch.randn(batch_size, 1, 101)
    x0 = x0.to('cpu')
    x1 = x1.to('cpu')
    x7 = x7.to('cpu')
    x8 = x8.to('cpu')
    y = model(x0, x1, x7, x8, 'cpu')