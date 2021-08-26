# encoding: utf-8
from transformers import BertTokenizer, MT5ForConditionalGeneration, MT5EncoderModel
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
import os, codecs, json, jieba, sys
from tqdm import tqdm
from rouge import Rouge
sys.setrecursionlimit(int(1e6))

# Set random seeds and deterministic pytorch for reproducibility
SEED = 1024 
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True
device = 'cuda' if cuda.is_available() else 'cpu'


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def load_model(tokenizer_path, model_path):
    print('tokenizer path:', tokenizer_path)
    print('model path:', model_path)
    tokenizer = T5PegasusTokenizer.from_pretrained(tokenizer_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    # test example
    text = '蓝蓝的天上有一朵白白的云'
    ids = tokenizer.encode(text, return_tensors='pt')
    print(ids)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    print(text)
    text = tokenizer.decode(ids[0], skip_special_tokens=False)
    print(text)
    model.eval()
    output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            max_length=30).numpy()[0]
    print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))
    model.to(device)
    return tokenizer, model


def load_dataset(path):
    dataset = []
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            data_f = json.loads(line)
            if len(data_f['answers']) <= 0: continue
            dataset.append((data_f['question'], data_f['answers'][0]))
        '''
        for item in data_f.values():
            trial_text = item["trial_text"].split("\n")
            trial_dialogue_orig = [utter.split("\t")[-1] for utter in trial_text]
            trial_dialogue = ""
            for utter in trial_dialogue_orig:
                trial_dialogue += "".join(utter.split())
            factfinding_text = "".join(item["factfinding_text"].split())
            dataset.append((trial_dialogue, factfinding_text, item["factfinding_text"]))
        '''
    return dataset


class CustomDataset(Dataset):

    def __init__(self, dialogue_data, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dialogue_data
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question_text = self.data[index][0]
        answer_text = self.data[index][1]

        question = self.tokenizer.batch_encode_plus([question_text], max_length= self.source_len, padding='max_length',return_tensors='pt', truncation=True)
        answer = self.tokenizer.batch_encode_plus([answer_text], max_length= self.summ_len, padding='max_length',return_tensors='pt', truncation=True)

        question_ids = question['input_ids'].squeeze()
        question_mask = question['attention_mask'].squeeze()
        answer_ids = answer['input_ids'].squeeze()
        answer_mask = answer['attention_mask'].squeeze()

        return {
            'question_ids': question_ids.to(dtype=torch.long), 
            'question_mask': question_mask.to(dtype=torch.long), 
            'answer_ids': answer_ids.to(dtype=torch.long),
            'answer_mask': answer_mask.to(dtype=torch.long),
            'answer_ids_y': answer_ids.to(dtype=torch.long), 
            'question_text': question_text,
            'answer_text': answer_text
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for iter, data in enumerate(tqdm(loader)):
        y = data['answer_ids'].to(device, dtype = torch.long)
        ids = data['question_ids'].to(device, dtype = torch.long)
        mask = data['question_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, labels=y)
        loss = outputs[0]
        
        if iter % 500 == 0:
            print(f'\nEpoch: {epoch}, Loss:  {loss.item()}')
            sys.stdout.flush()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(tokenizer, model, device, loader, summary_len, beam_size):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for iter, data in enumerate(tqdm(loader)):
            y = data['answer_ids'].to(device, dtype = torch.long)
            ids = data['question_ids'].to(device, dtype = torch.long)
            mask = data['question_mask'].to(device, dtype = torch.long)
            target = data['answer_text']

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=summary_len, 
                num_beams=beam_size,
                decoder_start_token_id = tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                early_stopping=True,
                repetition_penalty = 2.5,
                length_penalty = 1.0
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True).replace(' ', '') for g in generated_ids]
            if iter % 100 == 0:
                print(f'\nCompleted {iter}')
                sys.stdout.flush()

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Pretrained model path.")
parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path.")
parser.add_argument("--train_data", type=str, help="Training dataset path.")
parser.add_argument("--dev_data", type=str, help="Development dataset path.")
parser.add_argument("--test_data", type=str, help="Test dataset path.")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate.")
parser.add_argument("--bs", type=int, default=8, help="batch size.")
parser.add_argument("--epoch", type=int, default=50, help="Fine tuning epoch.")
parser.add_argument("--max_len", type=int, default=800, help="Article maximum length.") # word based
parser.add_argument("--summary_len", type=int, default=300, help="Summary length.")
parser.add_argument("--output", type=str, default="./results/pegasus/predictions.txt", help="Decode abstract summary file path.")
parser.add_argument("--reference", type=str, default="./results/pegasus/references.txt", help="Reference summary file path.")
parser.add_argument("--save_path", type=str, default="./saved/pegasus/")
parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Train mode or evaluate mode.")
parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding.")
parser.add_argument("--early_stop_num", type=int, default=10, help="Early stop number.")

args = parser.parse_args()
print(args)
tokenizer, model = load_model(args.tokenizer_path, args.model_path)
rouge = Rouge()

train_dataset = load_dataset(args.train_data)
val_dataset = load_dataset(args.dev_data)
test_dataset = load_dataset(args.test_data)
print(f"train dataset size: {len(train_dataset)}")
print(f"dev dataset size: {len(val_dataset)}")
print(f"test dataset size: {len(test_dataset)}")

# Creating the Training and Validation dataset for further creation of Dataloader
training_set = CustomDataset(train_dataset, tokenizer, args.max_len, args.summary_len)
val_set = CustomDataset(val_dataset, tokenizer, args.max_len, args.summary_len)
test_set = CustomDataset(test_dataset, tokenizer, args.max_len, args.summary_len)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': args.bs,
    'shuffle': True,
    'num_workers': 10
    }

val_params = {
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 10
    }

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)
test_loader = DataLoader(test_set, **val_params)

if args.mode == 'train':
    save_dir = os.path.join(args.save_path, f'pegasus_base_lr{args.lr}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    best_rouge_l, early_stop_num = 0, 0

    for iter in range(args.epoch):
        print("-"*50 + f"Epoch{iter}" + "-"*50)
        train(iter, tokenizer, model, device, training_loader, optimizer)

        print("-" * 50 + "Validation" + "-" * 50)
        predictions, actual = validate(tokenizer, model, device, test_loader, args.summary_len, args.beam_size)
        rouge_1, rouge_2, rouge_l, total = 0, 0, 0, len(actual)
        for pred, ref in zip(predictions, actual):
            rouge_scores = rouge.get_scores(pred, ref)
            rouge_1 += rouge_scores[0]['rouge-1']['f']
            rouge_2 += rouge_scores[0]['rouge-2']['f']
            rouge_l += rouge_scores[0]['rouge-l']['f']
        print('\nrouge 1:%.4f, rouge 2:%.4f, rouge l:%.4f' % (rouge_1/total, rouge_2/total, rouge_l/total))

        if rouge_l > best_rouge_l: 
            # save model
            model.save_pretrained(os.path.join(save_dir, f'iter{iter}'))
            #save to results
            with codecs.open(os.path.join(save_dir, f'iter{iter}', 'predictions.txt'), mode='w+', encoding='utf-8') as w_f:
                for predict in predictions:
                    w_f.write(predict+'\n')

            with codecs.open(os.path.join(save_dir, f'iter{iter}', 'references.txt'), mode='w+', encoding='utf-8') as w_f:
                for gold in actual:
                    w_f.write(gold+'\n')
            # set best rouge 
            best_rouge_l = rouge_l
            early_stop_num = 0
        else:
            early_stop_num += 1
        
        if early_stop_num >= args.early_stop_num:
            print("Early stopping...")
            exit()
        
else:
    print("-" * 50 + "Test" + "-" * 50)
    predictions, actual = validate(tokenizer, model, device, test_loader, args.summary_len, args.beam_size)

    #save to results
    with codecs.open(args.output, mode='w+', encoding='utf-8') as w_f:
        for predict in predictions:
            w_f.write(predict+'\n')

    with codecs.open(args.reference, mode='w+', encoding='utf-8') as w_f:
        for gold in actual:
            w_f.write(gold+'\n')