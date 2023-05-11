# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
from webbrowser import get
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from models.modeling_bert import BERTForTokenClassification_v2
from utils.data_utils import load_and_cache_examples, get_labels
from utils.model_utils import mask_tokens, soft_frequency, opt_grad, get_hard_label, _update_mean_model_variables
from utils.eval import evaluate
from utils.config import config
from utils.loss_utils import NegEntropy
from utils.pu_utils import my_pu_loss

logger = logging.getLogger(__name__)

MODEL_NAMES = {
    "student1":"Bert", 
    "student2":"Bert", 
    "teacher1":"Bert", 
    "teacher2":"Bert"
}
MODEL_CLASSES = {
    "student1": (BertConfig, BERTForTokenClassification_v2, BertTokenizer),
    "student2": (BertConfig, BERTForTokenClassification_v2, BertTokenizer),
}
LOSS_WEIGHTS = {
    "pseudo": 1.0,
    "self": 0.5,
    "mutual": 0.3,
    "mean": 0.2,
}
torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args,t_total,num_labels,epoch):
    config_class, model_class, _ = MODEL_CLASSES["student1"]
    config_s1 = config_class.from_pretrained(
        args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1 = model_class.from_pretrained(
        args.student1_model_name_or_path,
        from_tf=bool(".ckpt" in args.student1_model_name_or_path),
        config=config_s1,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_1 = [
        {
            "params": [p for n, p in model_s1.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_s1.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_s1 = AdamW(optimizer_grouped_parameters_1, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler_s1 = get_linear_schedule_with_warmup(
        optimizer_s1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    model_s1.zero_grad()

    return model_s1,  optimizer_s1, scheduler_s1, 


def initialize_from_one(args, t_total,num_labels, epoch):
    dir = os.path.join(args.output_dir,'../denoise/mooc/')
    path3 = os.path.join(dir,'student1','checkpoint-best-2')
    config_class, model_class, _ = MODEL_CLASSES["student1"]
    config_s1 = config_class.from_pretrained(
        args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1 = model_class.from_pretrained(path3,config=config_s1)
    model_s1.to(args.device)


    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_1 = [
        {
            "params": [p for n, p in model_s1.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_s1.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_s1 = AdamW(optimizer_grouped_parameters_1, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler_s1 = get_linear_schedule_with_warmup(
        optimizer_s1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    
  
    

    model_s1.zero_grad()




    return model_s1,  optimizer_s1, scheduler_s1,


def validation(args, model, tokenizer, labels, pad_token_label_id, best_dev, best_test, 
                  global_step, t_total, epoch):
    

    # results, _, best_dev, is_updated1 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev, mode="dev", \
    #     logger=logger, prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    results, _, best_test, is_updated2 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
        logger=logger, prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
   
    # output_dirs = []
    # notice we don't use checkpoint1 cause we have the same dev and test corpus
    # if args.local_rank in [-1, 0] and is_updated1:
    #     # updated_self_training_teacher = True
    #     path = os.path.join(args.output_dir+tors, "checkpoint-best-1")
    #     logger.info("Saving model checkpoint to %s", path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     model_to_save = (
    #             model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(path)
    #     tokenizer.save_pretrained(path)
    # # output_dirs = []
    if args.local_rank in [-1, 0] and is_updated2:
        # updated_self_training_teacher = True
        path = os.path.join(args.output_dir, "checkpoint-best")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
                model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return best_dev, best_test, is_updated2

def train(args, train_dataset, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    num_labels = len(labels)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    model, optimizer, scheduler = initialize(args, t_total,num_labels, 0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    self_training_teacher_model = model

    if args.pu_learning:
        pu_model = model

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # if global_step >= args.begin_step:
            #     delta = global_step-args.begin_step
            #     if delta%args.self_learning_period == 0:
            #         self_learning_teacher_model = copy.deepcopy(model)
            #         self_learning_teacher_model.eval()
                
            #     inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],'field_embeds':batch[4]}
            #     with torch.no_grad():
            #         outputs = self_learning_teacher_model(**inputs)
            #     label_mask = None
            #     if args.self_learning_label_mode == "soft":
            #         pred_labels = soft_frequency(logits=outputs[0], power=2)
            #         pred_labels, label_mask = mask_tokens(args,pred_labels,pad_token_label_id)

            #     inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2],'field_embeds':batch[4], "labels": {"pseudo": pred_labels}, "label_mask": label_mask}
            # else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2],'field_embeds':batch[4], "labels": {"pseudo": batch[3]}}     
            outputs = model(**inputs)

            loss = 0.0
            loss_dict = outputs[0]
            keys = loss_dict.keys()
            for key in keys:
                loss += LOSS_WEIGHTS[key]*loss_dict[key]
            # if epoch < args.begin_epoch:
                # loss1 += loss_regular(outputs1[1].view(-1, num_labels))
            
            pu_loss = 0.0

            # pu learning
            if args.pu_learning and epoch >= args.pu_begin_epoch:
                # update_step = global_step//args.pu_updatefreq
                # if update_step == 1:
                #     pu_model = copy.deepcopy(model)
                #     pu_model.train(True)
                # with torch.no_grad():
                #     pu_outputs = pu_model(**inputs)
                #     pu_loss = my_pu_loss(inputs['labels']['pseudo'],pu_outputs,args)
                pu_loss = my_pu_loss(inputs['labels']['pseudo'],outputs,args)

            #import pdb;pdb.set_trace()
            loss += args.pu_beta*pu_loss
            #pu_loss.backward()
            
            loss.backward()


            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info("***** Entropy loss: %.4f,",loss) 
                            
                        best_dev, best_test, _ = validation(args, model, tokenizer, labels, pad_token_label_id, \
                            best_dev, best_test, global_step, t_total, epoch)


                   
                   

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return global_step, tr_loss/global_step, results

def main():
    args = config()
    # args.do_train = args.do_train.lower()
    # args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    labels = get_labels(args.data_dir, args.dataset)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss, best_results = train(args,train_dataset, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

# def predict(args, tors, labels, pad_token_label_id, best_test):
#     path = os.path.join(args.output_dir+tors, "checkpoint-best-2")
#     tokenizer = RobertaTokenizer.from_pretrained(path, do_lower_case=args.do_lower_case)
#     model = RobertaForTokenClassification_Modified.from_pretrained(path)
#     model.to(args.device)

#     # if not best_test:
   
#     # result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
#     result, _, best_test, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
#                                                         logger=logger, verbose=False)
#     # Save results
#     output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
#     with open(output_test_results_file, "w") as writer:
#         for key in sorted(result.keys()):
#             writer.write("{} = {}\n".format(key, str(result[key])))

#     return best_test
#     # Save predictions
#     # output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
#     # with open(output_test_predictions_file, "w") as writer:
#     #     with open(os.path.join(args.data_dir, args.dataset+"_test.json"), "r") as f:
#     #         example_id = 0
#     #         data = json.load(f)
#     #         for item in data: # original tag_ro_id must be {XXX:0, xxx:1, ...}
#     #             tags = item["tags"]
#     #             golden_labels = [labels[tag] for tag in tags]
#     #             output_line = str(item["str_words"]) + "\n" + str(golden_labels)+"\n"+str(predictions[example_id]) + "\n"
#     #             writer.write(output_line)
#     #             example_id += 1

if __name__ == "__main__":
    main()
