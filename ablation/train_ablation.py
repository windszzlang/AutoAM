# train without stratified learning rate
from network import Network
from dataloader import get_dataloader
from utils import *

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = None
# seed = 667
dataset = 'CDCP'
# dataset = 'PE'
dataset_path = './data/' + dataset
save_path = './models/saved/best.pt'
PLM = 'roberta-base'

AC_type = {
    'CDCP': ['value', 'policy', 'testimony', 'fact', 'reference'],
    'PE': ['MajorClaim', 'Claim', 'Premise']
}
AR_type = {
    'CDCP': [ 'none', 'reason', 'evidence'],
    'PE': ['none', 'support', 'attack']
}


if seed != None:
    seed_everything(seed)



def train(epochs=50, min_epochs=10, patience=5, lr=1e-5, batch_size=24, max_len=512, device='cuda'):
    
    print('Preparing data......')
    id2label_comp, label2id_comp, num_labels_comp = get_id_label_map(AC_type[dataset])
    id2label_rel, label2id_rel, num_labels_rel = get_id_label_map(AR_type[dataset])
    print(id2label_comp)
    print(id2label_rel)

    tokenizer = AutoTokenizer.from_pretrained(PLM)

    train_data = process_data(load_data(os.path.join(dataset_path, 'train.jsonl'), tokenizer), label2id_comp, label2id_rel)
    # val_data = process_data(load_data(os.path.join(dataset_path, 'val.jsonl'), tokenizer), label2id_comp, label2id_rel)
    val_data = process_data(load_data(os.path.join(dataset_path, 'test.jsonl'), tokenizer), label2id_comp, label2id_rel)
    test_data = process_data(load_data(os.path.join(dataset_path, 'test.jsonl'), tokenizer), label2id_comp, label2id_rel)

    train_dataloader = get_dataloader(train_data, batch_size, max_len, label2id_comp, label2id_rel, tokenizer, device, is_shuffle=True)
    val_dataloader = get_dataloader(val_data, batch_size, max_len, label2id_comp, label2id_rel, tokenizer, device)
    test_dataloader = get_dataloader(test_data, batch_size, max_len, label2id_comp, label2id_rel, tokenizer, device)

    print('Building model......')
    bert  = AutoModel.from_pretrained(PLM)
    model = Network(bert, num_labels_comp, num_labels_rel)
    model = model.to(device)
    epsilon = 10
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': lr},
        {'params': model.attention.parameters(), 'lr': lr},
        {'params': model.comp_classifier.parameters(), 'lr': lr},
        {'params': model.rel_classifier.parameters(), 'lr': lr},
        {'params': model.distance_matrix.parameters(), 'lr': lr}
        # {'params': model.attention.parameters(), 'lr': epsilon * lr},
        # {'params': model.comp_classifier.parameters(), 'lr': epsilon * lr},
        # {'params': model.rel_classifier.parameters(), 'lr': epsilon * epsilon * lr},
        # {'params': model.distance_matrix.parameters(), 'lr': epsilon * epsilon * lr}
    ], lr=epsilon * lr)


    print('Start Training............')
    patience_cnt = 0
    best_perform = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0
        cnt_train = 0
        train_bar = tqdm(train_dataloader, position=0, leave=True)
        for batch_data in train_bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            cnt_train += 1
            train_bar.set_description(f'epoch {epoch}')
            train_bar.set_postfix(loss=train_loss / cnt_train)

        # print(f'Epoch [{epoch}/{epochs}]: loss: {train_loss / cnt_train}')
        
        ## valid
        pred_comp, gold_comp, pred_rel, gold_rel = [], [], [], []
        model.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_dataloader, position=0, leave=True)
            for batch_data in valid_bar:
                res = model.predict(batch_data)
                pred_comp.extend(res['pred_comp'])
                gold_comp.extend(res['gold_comp'])
                pred_rel.extend(res['pred_rel'])
                gold_rel.extend(res['gold_rel'])
                valid_bar.set_description(f'epoch {epoch}')

        scores_comp, scores_rel_bool, scores_rel_type, f1_score_avg = print_performance(pred_comp, gold_comp, pred_rel, gold_rel, test_type='val')
        cur_perform = scores_rel_bool['f1_score']
        # cur_perform = scores_comp['f1_score']
        print(f'Epoch [{epoch}/{epochs}]: scores_rel_bool_f1_score: {cur_perform}')
   
        if cur_perform > best_perform:
            patience_cnt = 0
            best_perform = cur_perform
            best_epoch = epoch
            # checkpoint = {
            #     'net': model.state_dict(),
            #     'optimizer':optimizer.state_dict(),
            #     'epoch': epoch
            # }
            # torch.save(checkpoint, save_path)
            torch.save(model, save_path)
            print('***** new score *****')
            print(f'The best epoch is: {best_epoch}, with the best performance is: {best_perform}')
            print('********************')
        elif patience_cnt >= patience and epoch > min_epochs: # early stop
            print(f'Early Stop with best epoch {best_epoch}, with the best performance is: {best_perform}')
            break
        if epoch > min_epochs:
            patience_cnt += 1


    ## test
    test_model = torch.load(save_path)
    pred_comp, gold_comp, pred_rel, gold_rel = [], [], [], []
    test_model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, position=0, leave=True)
        for batch_data in test_bar:
            res = test_model.predict(batch_data)
            pred_comp.extend(res['pred_comp'])
            gold_comp.extend(res['gold_comp'])
            pred_rel.extend(res['pred_rel'])
            gold_rel.extend(res['gold_rel'])

    scores_comp, scores_rel_bool, scores_rel_type, f1_score_avg = print_performance(pred_comp, gold_comp, pred_rel, gold_rel, test_type='test')
    test_perform = scores_rel_bool['f1_score']
    print(f'Performance on test dataset, scores_rel_bool_f1_score: {test_perform}')
    return model


if __name__ == '__main__':
    model = train(epochs=50, min_epochs=15, patience=5, lr=2e-5, batch_size=16, max_len=512, device='cuda') # CDCP
    # model = train(epochs=50, min_epochs=15, patience=5, lr=2e-5, batch_size=2, max_len=512, device='cuda') # PE