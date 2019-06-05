This is the code for replicating the language experiments. 

Requirements
---
- Python 3 with standard packages (e.g., numpy).
- PyTorch 1.0.
- [huggingface PyTorch implementation of Google AI's BERT model.](https://github.com/huggingface/pytorch-pretrained-BERT)
- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

Usage
---
1. Run `convert_lmrd.py` to convertthe Large Movie Review Dataset into formats the BERT models can read and shuffle them.
In our experiments, we limited the size of the test set to 1000 reviews.

`> cd scripts`

`> python convert_lmrd.py --input_dir [lmrd/download] --output_dir [lmrd_for_bert] --test_output_limit 1000`

2. Fine-tune a BERT language model on LMRD. The tuned model will live at `[lm_tune_out]/tuned_epoch_2.bin`

`> mv huggingface/run_lm_finetuning.py [huggingface]/examples/`

`> cd [huggingface]/examples/`

`> python run_lm_finetuning.py --bert_model bert-base-cased --do_train --train_file [lmrd_for_bert]/train.lang.txt --eval_file [lmrd_for_bert]/test.lang.txt --output_dir [lm_tune_out] --max_seq_length 128 --num_train_epochs 3`

3. Train a BERT for sequence classification model on LMRD.
The trained model will live at `pred_tune_out/pytorch_model.bin`, and the output json for the experiment will live at `[exp_outfile]`

`> mv huggingface/run_classifier.py [huggingface]/examples/`

`> cd [huggingface]/examples/`

`> python3 run_classifier.py --data_dir [lmrd_for_bert] --bert_model bert-base-cased --task_name lmrd --output_dir [pred_tune_out] --max_seq_length 256 --do_train --do_eval --num_train_epochs 2 --do_mask_eval --mask_eval_outfile [exp_outfile] --input_lm_model_file [lm_tune_out]/tuned_epoch_2.bin`

4. Generate sample review outputs with WordPiece's whose t values allow us to reject the null hypothesis highlighted.

`> cd scripts`

`> python fdr_lang.py --input_fn [exp_outfile] --nlabels 2 --test directionless --alpha 0.15 --output_dir [analysis_out]`
