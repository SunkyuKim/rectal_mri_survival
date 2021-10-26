import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import logging
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertModel


class PartialNLL(nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    def forward(self, theta, R, censored):
        theta = theta.double()
        exp_theta = torch.exp(theta)

        observed = 1-censored
        num_observed = torch.sum(observed)
        #loss = -(torch.sum((theta.reshape(-1)- torch.log(torch.sum((exp_theta * R.t()), 0))) * observed) / num_observed)
        loss = -(torch.sum((theta.reshape(-1)- torch.log(torch.sum((exp_theta * R.t()), 0))) * observed))

        if np.isnan(loss.data.tolist()):
            for a,b in zip(theta, exp_theta):
                print("Loss is nan", a.tolist(),b.tolist())
            exit()

        return loss

class PosNegAttnLSTMCox(nn.Module):
    def __init__(self, vocab_size):
        super(PosNegAttnLSTMCox, self).__init__()

        emb = 200
        hidden = 100
        attnhidden = 100
        bidirectional = True

        dim_demo_in = 158
        dim_demo_out = 200

        self.embedding = nn.Embedding(vocab_size, emb)

        #self.lstm = nn.LSTM(emb, hidden, bidirectional=bidirectional)
        self.lstm = nn.GRU(emb, hidden, bidirectional=bidirectional)

        self.attn_pos1 = nn.Linear(hidden*2, attnhidden)
        self.attn_pos2 = nn.Linear(attnhidden, 1)
        self.context_bn_pos = nn.BatchNorm1d(hidden*2)

        self.attn_neg1 = nn.Linear(hidden*2, attnhidden)
        self.attn_neg2 = nn.Linear(attnhidden, 1)
        self.context_bn_neg = nn.BatchNorm1d(hidden*2)


        self.demo = nn.Linear(dim_demo_in, dim_demo_out)
        self.demo_bn = nn.BatchNorm1d(dim_demo_out)

        self.dropout = nn.Dropout(p=0.2)

        n_layers = 2

        fcs = list()
        for i in range(n_layers):
            fcs.append(nn.Linear(hidden*2, hidden*2))
            fcs.append(nn.BatchNorm1d(hidden*2))
            fcs.append(nn.ReLU())
        fcs.append(nn.Linear(hidden*2, 1))

        self.fc = nn.ModuleList(fcs)

    def forward(self, sent, sent_len, demo):
        embeds = self.embedding(sent)

        embeds = nn.utils.rnn.pack_padded_sequence(embeds, sent_len, batch_first=True)

        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out_arranged = lstm_out.view(len(sent), -1, 200) #(batch, seq, hidden)

        # Lin et al 2017, attention for sentence classification
        attn_pos1 = self.attn_pos1(lstm_out_arranged) #(batch, seq, atthidden)
        attn_pos1 = torch.tanh(attn_pos1)
        att_pos_w = self.attn_pos2(attn_pos1) #(batch, seq, 1)
        softmax_att_pos_w = F.softmax(att_pos_w, 1).squeeze(2) #(batch, seq)
        context_pos = torch.bmm(lstm_out_arranged.transpose(1,2), softmax_att_pos_w.unsqueeze(2)).squeeze(2) #(batch, hidden)


        # Lin et al 2017, attention for sentence classification
        attn_neg1 = self.attn_neg1(lstm_out_arranged) #(batch, seq, atthidden)
        attn_neg1 = torch.tanh(attn_neg1)
        att_neg_w = self.attn_neg2(attn_neg1) #(batch, seq, 1)
        softmax_att_neg_w = F.softmax(att_neg_w, 1).squeeze(2) #(batch, seq)
        context_neg = torch.bmm(lstm_out_arranged.transpose(1,2), softmax_att_neg_w.unsqueeze(2)).squeeze(2) #(batch, hidden)


        demo_out = torch.relu(self.demo(demo))

        context_pos = context_pos * demo_out
        context_neg = context_neg * demo_out


        for l in self.fc:
            context_pos = l(context_pos)
        context_pos = torch.relu(context_pos)

        for l in self.fc:
            context_neg = l(context_neg)
        context_neg = torch.relu(context_neg)

        out = context_pos - context_neg

        att = torch.stack((softmax_att_pos_w, softmax_att_neg_w), 2)
        context = torch.stack((context_pos, context_neg), 2)

        return out, att, context



class AttnLSTMCox(nn.Module):
    def __init__(self, vocab_size):
        super(AttnLSTMCox, self).__init__()

        emb = 128
        hidden = 128
        attnhidden = 128
        bidirectional = True

        dim_demo_in = 158
        dim_demo_out = 256

        self.embedding = nn.Embedding(vocab_size, emb)

        #self.lstm = nn.LSTM(emb, hidden, bidirectional=bidirectional)
        self.lstm = nn.GRU(emb, hidden, bidirectional=bidirectional)

        self.attn1 = nn.Linear(hidden*2, attnhidden)
        self.attn2 = nn.Linear(attnhidden, 1)
        self.context_bn = nn.BatchNorm1d(hidden*2)

        self.demo = nn.Linear(dim_demo_in, dim_demo_out)
        self.demo_bn = nn.BatchNorm1d(dim_demo_out)

        self.dropout = nn.Dropout(p=0.2)

        n_layers = 1

        fcs = list()
        for i in range(n_layers):
            fcs.append(nn.Linear(hidden*2, hidden*2))
            fcs.append(nn.BatchNorm1d(hidden*2))
            fcs.append(nn.ReLU())
        fcs.append(nn.Linear(hidden*2, 1))

        self.fc = nn.ModuleList(fcs)

    def forward(self, sent, demo=None, attention_mask=None):

        if attention_mask is not None:
            batch_x_len = attention_mask.sum(1)
            batch_x_len, sorted_idx = batch_x_len.sort(0, descending=True)
            _, origin_idx = sorted_idx.sort(0, descending=False)
            sent = sent[sorted_idx]

            embeds = self.embedding(sent)
            embeds = nn.utils.rnn.pack_padded_sequence(embeds, batch_x_len, batch_first=True)

            lstm_out, _ = self.lstm(embeds)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out_arranged = lstm_out.view(len(sent), -1, 256) #(batch, seq, hidden)
            lstm_out_arranged = lstm_out_arranged[origin_idx]

        # Lin et al 2017, attention for sentence classification
        attn1 = self.attn1(lstm_out_arranged) #(batch, seq, atthidden)
        attn1 = torch.tanh(attn1)
        att_w = self.attn2(attn1) #(batch, seq, 1)

        #final_lstm_out = lstm_out[:,-1,:].view(len(sent), 200, 1) #(batch, hidden, 1)
        #att_w = torch.bmm(lstm_out_arranged, final_lstm_out) #(batch, seq, 1)

        softmax_att_w = F.softmax(att_w, 1).squeeze(2) #(batch, seq)

        context = torch.bmm(lstm_out_arranged.transpose(1,2), softmax_att_w.unsqueeze(2)).squeeze(2) #(batch, hidden)

        #context = self.dropout(context)

        if demo is not None:
            demo_out = torch.relu(self.demo(demo))
            context = context * demo_out

        for l in self.fc:
            context = l(context)
        out = context

        return out, softmax_att_w, context

#    def train_step(self):
#        pass
#
#    def eval_step(self):
#        pass


class RETAINCox(nn.Module):
    def __init__(self, vocab_size):
        super(RETAINCox, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 100)

        self.alpha_lstm = nn.RNN(100, 1)
        self.beta_lstm = nn.RNN(100, 100)

        self.fc = nn.Linear(100, 1)

    def forward(self, sent, *args):
        embeds = self.embedding(sent)

        inverse_idx = torch.arange(embeds.shape[1]-1, -1, -1).long()
        embeds_inverse = embeds.index_select(1, inverse_idx)

        alpha_out, _ = self.alpha_lstm(embeds_inverse)
        beta_out, _ = self.beta_lstm(embeds_inverse)

        alpha_out = alpha_out.index_select(1, inverse_idx)
        beta_out = beta_out.index_select(1, inverse_idx)

        context = alpha_out*beta_out
        context = context*embeds

        context_sum = context.sum(1)
        out = self.fc(context_sum)
        return out, alpha_out, out


class BertCox(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #self.num_labels = config.num_labels
        self.num_labels = 1

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = False

        attnhidden = 200
        self.attn1 = nn.Linear(config.hidden_size, attnhidden)
        self.attn2 = nn.Linear(attnhidden, 1)
        self.context_bn = nn.BatchNorm1d(config.hidden_size)

        dim_demo_in = 158
        self.demo = nn.Linear(dim_demo_in, config.hidden_size)
        self.demo_bn = nn.BatchNorm1d(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.fc = nn.Linear(config.hidden_size, self.num_labels)

        n_layers = 1

        fcs = list()
        for i in range(n_layers):
            fcs.append(nn.Linear(config.hidden_size, config.hidden_size))
            fcs.append(nn.BatchNorm1d(config.hidden_size))
            #fcs.append(nn.ReLU())
            fcs.append(nn.Tanh())
        self.fc = nn.ModuleList(fcs)

        self.final = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        demo=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        if attention_mask is None:
            print("None??")
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
#        print(input_ids)
#        print(outputs[0])
#        print(attention_mask)
#        print(attention_mask.sum(1))
#        print((~attention_mask).sum(1))
        context = outputs[0]
        attn1 = self.attn1(context) #(batch, seq, atthidden)
        attn1 = torch.tanh(attn1)
        att_w = self.attn2(attn1).squeeze(2) #(batch, seq, 1)
        att_w[~attention_mask] = float('-inf')
        softmax_att_w = F.softmax(att_w, 1) #(batch, seq)
        context = torch.bmm(context.transpose(1,2), softmax_att_w.unsqueeze(2)).squeeze(2) #(batch, hidden)

        text_context = context

        if demo is not None:
            #demo_out = torch.relu(self.demo(demo)
            demo_out = torch.tanh(self.demo(demo))
            context = context + demo_out

        context = self.context_bn(context)
        #context = self.dropout(context)

        for l in self.fc:
            context = l(context)

        outputs = self.final(context)

        return outputs, softmax_att_w, text_context



class VisitBertCox(BertPreTrainedModel):
    def __init__(self, config, dim_demo_in, only_first_report):
        super().__init__(config)
        self.num_labels = 1
        self.only_first_report = only_first_report

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = False

        #self.gru = nn.GRU(768, 768, batch_first=True)
        self.gru = nn.GRUCell(config.hidden_size, config.hidden_size)

        attnhidden = 200
        self.attn1 = nn.Linear(config.hidden_size, attnhidden)
        self.attn2 = nn.Linear(attnhidden, 1)
        self.context_bn = nn.BatchNorm1d(config.hidden_size)

        #dim_demo_in = dim_demo
        self.demo = nn.Linear(dim_demo_in, config.hidden_size)
        self.demo_bn = nn.BatchNorm1d(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

        self.submulnn = nn.Sequential(
             nn.Linear(config.hidden_size*2, config.hidden_size),
             nn.ReLU(),
             nn.Linear(config.hidden_size, config.hidden_size),
             nn.ReLU(),
             nn.Linear(config.hidden_size, config.hidden_size),
             nn.Tanh()
             )

        self.final = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        demo=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,

    ):
        if attention_mask is None:
            print("None??")

        input_ids_list = input_ids.unbind(1) # (batch, visitlen, seqlen) -> visitlen * (batch, seqlen)
        demo_list = demo.unbind(1)
        outputs_list = list()

        gru_h = None
        for i in range(len(input_ids_list)):
            input_ids = input_ids_list[i] # (batch, seqlen)
            demo_vec = demo_list[i]

            attention_mask = (input_ids>0).byte()
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            last_seq_out = outputs[0][:,-1,:]
            gru_h = self.gru(last_seq_out, gru_h)

#            demo_out = self.demo(demo_vec)
#
#            # SubMilt+NN comparison, Wang and Jiang
#            sub = (gru_h-demo_out)*(gru_h-demo_out)
#            mul = gru_h*demo_out
#            output = self.submulnn(torch.cat([sub,mul],1))

            output = gru_h
            outputs_list.append(output)

            if self.only_first_report:
                break # use only first visit report


        theta_list = list()
        context = outputs_list[0] + outputs_list[-1]
        context = self.fc(context)
        context = torch.tanh(context)
        theta = self.final(context)
        theta_list.append(theta)

        return theta_list, context, context


if __name__ == "__main__":
    pass
