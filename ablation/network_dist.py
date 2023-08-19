# train without distance matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super().__init__()
        self.w_query = nn.Linear(input_size, attention_size, bias=False)
        self.w_key = nn.Linear(input_size, attention_size, bias=False)
        self.w_value = nn.Linear(input_size, attention_size, bias=False)
        self.dimension_k = math.sqrt(attention_size)

    def forward(self, hidden_state):
        '''hidden_state dim=[num_rel, bert_hidden_size]
        '''
        Q = self.w_query(hidden_state) # dim=[num_rel, attention_size]
        K = self.w_key(hidden_state).transpose(-1, -2) # dim=[attention_size, num_rel]
        V = self.w_value(hidden_state) # dim=[num_rel, attention_size]
        atten_score = F.softmax(torch.matmul(Q, K) / self.dimension_k, dim=-1) # dim=[num_rel, num_rel]
        out = torch.matmul(atten_score, V) # dim=[num_rel, attention_size]
        return out


class Network(nn.Module):
    def __init__(self, bert_model, num_labels_comp, num_labels_rel):
        super().__init__()
        self.bert = bert_model
        self.bert_config = self.bert.config
        self.num_labels_comp = num_labels_comp
        self.num_labels_rel = num_labels_rel
        self.dropout = nn.Dropout(0.2)
        self.comp_classifier = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, num_labels_comp)
        )

        self.attention_size = self.bert_config.hidden_size
        self.attention = Attention(self.bert_config.hidden_size, self.attention_size)
        
        self.layer_norm = nn.LayerNorm(self.attention_size)
        # self.dist_size = self.bert_config.hidden_size
        # self.distance_matrix = nn.Linear(1, self.dist_size)
        
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size * 2, 512),
            # nn.Linear(self.bert_config.hidden_size * 2 + self.dist_size, 512),
            nn.Tanh(),
            nn.Linear(512, num_labels_rel)
        )
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.argmax_threshold = 0.05
        

    def forward_comp_cls(self, input_ids, attention_mask, spans, components=None):
        last_hidden_state = self.bert(input_ids, attention_mask,
                                    return_dict=True)['last_hidden_state'] # dim=[batch_size, seq_len, hidden_size]
        pred, logits, bert_output_comp = [], [], []
        for i, tmp_spans in enumerate(spans):
            tmp_pred = []
            tmp_output_comp = []
            for start, end in tmp_spans:
                # ACs = [start, end] in Text
                comp = last_hidden_state[i, start:end+1, :].squeeze(0) # dim=[span_len, hidden_size]
                comp = torch.mean(comp, dim=0, keepdim=False) # mean pooling, dim=[hidden_size]
                comp = self.dropout(comp)
                # print(start, end, last_hidden_state.size())
                # print(comp)
                cls_out = self.comp_classifier(comp) # dim=[num_labels]
                logits.append(cls_out)
                tmp_pred.append(torch.argmax(F.softmax(cls_out, dim=0), keepdim=False))
                tmp_output_comp.append(comp)
            tmp_output_comp = torch.stack(tmp_output_comp, dim=0) # dim=[num_comp, hidden_size]
            atten_out = self.attention(tmp_output_comp) # dim=[num_comp, hidden_size]
            tmp_output_comp = self.layer_norm(tmp_output_comp + atten_out) # dim=[num_comp, hidden_size]
            pred.append(tmp_pred)
            bert_output_comp.append(tmp_output_comp)
        if components == None:
            return pred, None, bert_output_comp
        logits = torch.stack(logits, dim=0) # dim=[num_comp, num_labels]
        gold = torch.cat(components, dim=0).long() # dim=[num_comp]
        loss = self.criterion(logits, gold)
        return pred, loss, bert_output_comp


    def forward_rel_cls(self, bert_output_comp, relations=None):
        pred, logits = [], []
        for comps in bert_output_comp:
            rel_hidden_state = []
            tmp_pred = []
            for i in range(len(comps)):
                for j in range(len(comps)):
                    if i == j:
                        continue
                    # distance = torch.Tensor([i - j]).type_as(comps[i])
                    # dist_emb = self.distance_matrix(distance)
                    # comp_pair = torch.cat([comps[i], comps[j], dist_emb], dim=0) # dim=[hidden_size * 3]
                    comp_pair = torch.cat([comps[i], comps[j]], dim=0) # dim=[hidden_size * 3]
                    rel_hidden_state.append(comp_pair)
            rel_hidden_state = torch.stack(rel_hidden_state, dim=0) # dim=[num_rel, hidden_size * 3]
            for i in range(rel_hidden_state.size(0)):
                cls_out = self.rel_classifier(rel_hidden_state[i, :]) # dim=[num_labels]
                logits.append(cls_out)
                tmp_pred.append(F.softmax(cls_out, dim=0)) # do not argmax now
            pred.append(tmp_pred)
        if relations == None:
            return pred, None
        logits = torch.stack(logits, dim=0) # dim=[num_rel, num_labels]
        gold = torch.cat(relations, dim=0).long()
        loss = self.criterion(logits, gold)
        return pred, loss


    def forward(self, input_ids, attention_mask, spans, components=None, relations=None):
        pred_comp, loss_comp, bert_output_comp = self.forward_comp_cls(input_ids, attention_mask, spans, components)
        pred_rel, loss_rel = self.forward_rel_cls(bert_output_comp, relations)
        return pred_comp, pred_rel, loss_comp, loss_rel


    def compute_loss(self, batch_data):
        self.train()
        _, _, loss_comp, loss_rel = self(batch_data['encodings']['input_ids'], batch_data['encodings']['attention_mask'], batch_data['ACs_span'], batch_data['ACs'], batch_data['ARs'])
        loss = loss_comp + loss_rel
        return loss


    def predict(self, batch_data):
        self.eval()
        pred_comp, pred_rel, _, _ = self(batch_data['encodings']['input_ids'], batch_data['encodings']['attention_mask'], batch_data['ACs_span'], batch_data['ACs'], batch_data['ARs'])
        pred_comp = [q for p in pred_comp for q in p]
        gold_comp = [comp for comps in batch_data['ACs'] for comp in comps]
        pred_rel = [q for p in pred_rel for q in p]
        gold_rel = [rel for rels in batch_data['ARs'] for rel in rels]
        self.train()
        return {'pred_comp': pred_comp, 'gold_comp': gold_comp, 'pred_rel': pred_rel, 'gold_rel': gold_rel}
 

    # def threshold_argmax(self, cls_out, keepdim=False):
    #     for i in range(1, len(cls_out)):
    #         if cls_out[i] > self.argmax_threshold:
    #             cls_out[0] = 0
    #             break
    #     return torch.argmax(cls_out, keepdim=keepdim)