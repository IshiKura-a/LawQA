CUDA_VISIBLE_DEVICES=3 python models/pegasus.py \
   --model_path '/data/home/ganleilei/bert/chinese_t5_pegasus_small/' \
   --tokenizer_path '/data/home/ganleilei/bert/chinese_t5_pegasus_small/' \
   --train_data '/data/home/ganleilei/law/law-qa/test_qa_corpus.json' \
   --dev_data  '/data/home/ganleilei/law/law-qa/dev_qa_corpus.json' \
   --test_data '/data/home/ganleilei/law/law-qa/test_qa_corpus.json' \
   --save_path '/data/home/ganleilei/outputs/law-qa/saved'