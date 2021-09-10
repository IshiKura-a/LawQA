from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import argparse, codecs, sys
from tqdm import tqdm
from bert_score import BERTScorer
import jieba
sys.setrecursionlimit(int(1e6))

parser = argparse.ArgumentParser()
parser.add_argument('--predictions', type=str, help='Decode text.')
parser.add_argument('--reference', type=str, help='Gold text.')
parser.add_argument('--cut', action='store_true', help='Whether cut the sentence')
parser.add_argument('--model_path', type=str, help='Bert model path.')

args = parser.parse_args()

rouge = Rouge()
smooth_func = SmoothingFunction().method1

bert_scorer = BERTScorer(lang="zh", batch_size=4, device='cuda:0', rescale_with_baseline=False, model_type=args.model_path, num_layers=12)

predictions, reference = [], []
with codecs.open(args.predictions, mode='r', encoding='utf-8') as rf:
    for line in rf.readlines():
        predictions.append(line.strip())

with codecs.open(args.reference, mode='r', encoding='utf-8') as rf:
    for line in rf.readlines():
        reference.append(line.strip())

(P, R, F), hash_tag = bert_scorer.score(predictions, reference, return_hash=True)

print(f"{hash_tag} P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")

rouge_1, rouge_2, rouge_l = 0, 0, 0
bleuscore1, bleuscore2, bleuscoren = 0, 0, 0

for pred, ref in tqdm(zip(predictions, reference)):
    if args.cut:
            pred = ' '.join(list(jieba.cut(pred)))
            reference = ' '.join(list(jieba.cut(ref)))
            
    scores = rouge.get_scores(pred, ref)
    rouge_1 += scores[0]['rouge-1']['f']
    rouge_2 += scores[0]['rouge-2']['f']
    rouge_l += scores[0]['rouge-l']['f']
    #bleu += sentence_bleu(ref.split(' '), pred.split(' '), smoothing_function=smooth_func)
    #bleuscore1 += sentence_bleu(ref, pred, weights=(1, 0, 0, 0))
    #bleuscore2 += sentence_bleu(ref, pred, weights=(0, 1, 0, 0))
    #bleuscoren += sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25))


total = len(predictions)

print('rouge_1:', rouge_1/total)
print('rouge_2:', rouge_2/total)
print('rouge_l:', rouge_l/total)
#print("bleu 1:", bleuscore1/total)
#print("bleu 2:", bleuscore2/total)
#print("bleu n:", bleuscoren/total)