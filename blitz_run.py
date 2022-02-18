# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# yuxiaw
# 24 April 2021
# For Regression uncertainty estimation using MC dropout
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from args import parse_arguments, default_arguments
from Non_Bayesian_models import HConvBertForSequenceClassification
from Bayesian_models import BBNBertForSequenceClassification, BBNHConvBertForSequenceClassification
from run_classifier_dataset_utils import (processors, output_modes, tasks_num_labels,
                                         convert_examples_to_features, compute_metrics)
from util_quantile import check_Z, summary, eval_uncertainty

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'hconvbert': (BertConfig, HConvBertForSequenceClassification, BertTokenizer),
    'blitzbert': (BertConfig, BBNBertForSequenceClassification, BertTokenizer),
    'blitzhconv': (BertConfig, BBNHConvBertForSequenceClassification, BertTokenizer),
}

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    # ^^ safe to call this function even if cuda is not available

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.weight_path_or_name.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, 
                                                label_list, 
                                                max_seq_length=args.max_seq_length,
                                                tokenizer=tokenizer, 
                                                output_mode=output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    # Set num_labels
    num_labels = tasks_num_labels[args.task_name]

    # Prepare data loader
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    # Prepare optimizer
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                bias_correction=False,
                                max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            if args.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif args.output_mode == "regression":
                loss_fct = MSELoss()
                # print(logits.size(), label_ids.size())
                # loss = loss_fct(logits.view(-1), label_ids.view(-1))
                loss = model.sample_elbo(outputs=logits.view(-1),
                           labels=label_ids.view(-1),
                           criterion=loss_fct,
                           sample_nbr=3,
                           complexity_cost_weight=1/input_ids.size()[0])

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', loss.item(), global_step)

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    ### Example:
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(args.device)

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, MC_dropout = False):
    ### Evaluation
    num_labels = tasks_num_labels[args.task_name]
    eval_data = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if not MC_dropout:
        model.eval()
    else:
        model.train()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    out_label_ids = None

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # create eval loss and other metric required by the task
        if args.output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif args.output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    result['eval_loss'] = eval_loss
    
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s" % (key, str(result[key])))
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return preds, out_label_ids


def blitz_eval(args, model, tokenizer, num_sampling):
    result, predictions = [], []
    for i in range(num_sampling):
        preds, out_label_ids = evaluate(args, model, tokenizer, MC_dropout = False)
        preds = preds.round(decimals=2)
        result.append(preds)
        # predictions += preds
    # save the predicted score in "preds.txt"
    # with open(os.path.join(args.output_dir, "preds.txt"), "w") as file:
    #     file.write("\n".join([str(i) for i in predictions]))
    # using result to calculate mean and deviation, matrix mean and deviation
    result = np.array(result)
    mean, std = result.mean(0), np.std(result, axis=0)
    mean = mean.round(decimals=6)
    std = std.round(decimals=6)
    R = compute_metrics(args.task_name, mean, out_label_ids)
    for key in sorted(R.keys()):
        logger.info("  %s = %s", key, str(R[key]))
        print("%s = %s" % (key, str(R[key])))
    # save mean and std for each instance
    with open(os.path.join(args.output_dir, "mean_std.txt"), "w") as file:
        file.write("\n".join([str(i) + "\t" + str(j) for i,j in zip(mean, std)]))

    # uncertainty metric calculation
    # samples >=30: make sure 0.05*30 >= 1, otherwise kthvalue get errors
    print(result.shape, mean.shape, std.shape) # assume [samples, N], [N,], [N,]
    # get y_pj by kthvalue of samples result, more practical
    y = summary({"obs": torch.tensor(result, dtype=torch.float)})['obs']
    uncertainty_metrics_y = eval_uncertainty(out_label_ids, y)
    # get y_pj by normal distribution Z table, more theoretical 
    mean = torch.tensor(mean, dtype=torch.float)
    std = torch.tensor(std, dtype=torch.float)
    preds = check_Z(mean, std)
    uncertainty_metrics_preds = eval_uncertainty(out_label_ids, preds)

def MC_droput(args, model, tokenizer, num_sampling):
    result, predictions = [], []
    for i in range(num_sampling):
        preds, out_label_ids = evaluate(args, model, tokenizer, MC_dropout = True)
        preds = preds.round(decimals=2)
        result.append(preds)
        # predictions += preds
    # save the predicted score in "preds.txt"
    # with open(os.path.join(args.output_dir, "preds.txt"), "w") as file:
    #     file.write("\n".join([str(i) for i in predictions]))
    # using result to calculate mean and deviation, matrix mean and deviation
    result = np.array(result)
    mean, std = result.mean(0), np.std(result, axis=0)
    mean = mean.round(decimals=2)
    std = std.round(decimals=2)
    R = compute_metrics(args.task_name, mean, out_label_ids)
    for key in sorted(R.keys()):
        logger.info("  %s = %s", key, str(R[key]))
        print("%s = %s" % (key, str(R[key])))
    # save mean and std for each instance
    with open(os.path.join(args.output_dir, "mean_std.txt"), "w") as file:
        file.write("\n".join([str(i) + "\t" + str(j) for i,j in zip(mean, std)]))

    # uncertainty metric calculation
    # samples >=30: make sure 0.05*30 >= 1, otherwise kthvalue get errors
    print(result.shape, mean.shape, std.shape) # assume [samples, N], [N,], [N,]
    # get y_pj by kthvalue of samples result, more practical
    y = summary({"obs": torch.tensor(result, dtype=torch.float)})['obs']
    uncertainty_metrics_y = eval_uncertainty(out_label_ids, y)
    # get y_pj by normal distribution Z table, more theoretical 
    mean = torch.tensor(mean, dtype=torch.float)
    std = torch.tensor(std, dtype=torch.float)
    preds = check_Z(mean, std)
    uncertainty_metrics_preds = eval_uncertainty(out_label_ids, preds)
    return result

def main():
    # load arguments
    args = parse_arguments(sys.argv)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(filename = args.logfile_dir,
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.weight_path_or_name, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.weight_path_or_name, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.weight_path_or_name, num_labels=num_labels)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model to device
    if args.fp16:
        model.half()
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # args setting done
    logger.info("Training/evaluation parameters %s", args)
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    # Training
    if args.do_train:
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        preds, _ = evaluate(args, model, tokenizer, MC_dropout = False)
    # Prediction Uncertainty Estimation
    if args.do_mcdropout:
        blitz_eval(args, model, tokenizer, num_sampling = args.num_mc_sampling)
        # result = MC_droput(args, model, tokenizer, num_sampling = args.num_mc_sampling)

if __name__ == "__main__":
    main()

