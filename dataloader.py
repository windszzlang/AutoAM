import torch
from torch.utils.data import Dataset, DataLoader



class DIYDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class DIYCollator():
    def __init__(self, max_len, label2id_ac, label2id_ar, tokenizer, device, is_predict=False):

        self.max_len = max_len
        self.label2id_ac = label2id_ac
        self.label2id_ar = label2id_ar
        self.tokenizer = tokenizer
        self.device = device
        self.is_predict = is_predict

    # list -> dict: [[feature_1, feature_2], [feature_1, feature_2]] -> {'feature_1': [x, x], 'feature_2': [x, x]}
    def __call__(self, batch_data):
        res = dict()
        texts = [D['text'] for D in batch_data]
        res['encodings'] = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding='longest',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True # get offset_mapping by tokenizer
        ).to(self.device)
        # test: find longer sequence (disable truncation first)
        # for i, text in zip(res['encodings']['input_ids'], texts):
        #     if len(i) > 512:
        #         print(len(i))
        #         print(text)
        offset_mapping = res['encodings']['offset_mapping']
        res['ACs_span'] = [self.update_spans(D['ACs_span'], offset_mapping[i]) for i, D in enumerate(batch_data)]
        # res['ACs_span'] = [self.update_spans(D['ACs_span'], D['text']) for D in batch_data]
        if self.is_predict:
            return res

        # overlength cases
        for i in range(len(res['ACs_span'])):
            ACs_num = len(res['ACs_span'][i])
            new_ARs = []
            c = -1
            for a in range(len(batch_data[i]['ACs'])):
                for b in range(len(batch_data[i]['ACs'])):
                    if a == b:
                        continue
                    c += 1
                    if a < ACs_num and b < ACs_num:
                        # print(c, ACs_num, len(batch_data[i]['ACs']))
                        AR = batch_data[i]['ARs'][c]
                        new_ARs.append(AR)
            batch_data[i]['ARs'] = new_ARs
            batch_data[i]['ACs'] = batch_data[i]['ACs'][:ACs_num]


        res['ACs'] = [torch.tensor(D['ACs'], device=self.device) for D in batch_data]
        res['ARs'] = [torch.tensor(D['ARs'], device=self.device) for D in batch_data]

        return res

    # spans of character -> spans of token
    # span gaps exist in PE dataset, and need to be dealed with
    # offset_mapping: [[token_start, token_end]...]
    def update_spans(self, spans, mapping):
        new_spans = []
        # mapping = [char_start, char_end], [CLS]/[SEP] = (0,0)
        s_i = 0
        tmp_new_span_l, tmp_new_span_r = 0, 0
        find_l = False
        for i, (start, end) in enumerate(mapping):
            if s_i >= len(spans):
                break
            if start == end == 0:
                continue
            span = spans[s_i]
            if start <= span[0] <= end and not find_l:
                tmp_span_l = i
                find_l = True
            if start <= span[1] <= end:
                tmp_span_r = i
                new_spans.append([tmp_span_l, tmp_span_r])
                s_i += 1
                find_l = False
        # overlength cases
        # for i in range(s_i, len(spans)):
            # new_spans.append([512, 512])
        return new_spans


def get_dataloader(data, batch_size, max_len, label2id_ac, label2id_ar, tokenizer, device, is_shuffle=False, is_predict=False):
    convert_to_features = DIYCollator(max_len, label2id_ac, label2id_ar, tokenizer, device, is_predict)
    dataset = DIYDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle, # reshuffled at every epoch
        collate_fn=convert_to_features
    )
    return dataloader



    # old wrong version
    # def update_spans(self, spans, text):
    #     # span gaps exist in PE dataset, and need to be dealed with
    #     split_text = []
    #     is_gap = []
    #     last_end = -1
    #     for start, end in spans:
    #         # if start - last_end > 5: # xx.\n\nxxx in PE
    #         split_text.append(text[last_end+1:start])
    #         is_gap.append(1)
    #         split_text.append(text[start:end+1])
    #         is_gap.append(0)
    #         last_end = end

    #     encodings = self.tokenizer(
    #         split_text,
    #         add_special_tokens=False,
    #         max_length=self.max_len,
    #         truncation=True,
    #         return_tensors=None,
    #         return_length=True
    #     )

    #     candidate_spans = []
    #     last_pos = 1 # [CLS] at begin
    #     for length in encodings['length']:
    #         # overlength cases
    #         # if last_pos >= 512:
    #             # candidate_spans.append([512, 512])
    #         candidate_spans.append([last_pos, last_pos + length - 1]) # [start, end]
    #         last_pos += length
    #     # filter gap texts
    #     new_spans = []
    #     for span, g in zip(candidate_spans, is_gap):
    #         if not g:
    #             new_spans.append(span)
    #     return new_spans