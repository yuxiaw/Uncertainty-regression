import argparse
from argparse import Namespace
import time
import sys

# pass arguments by command line or a config.txt/.json file
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--weight_path_or_name", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_type",
                        default="bert",
                        type=str,
                        required=True,
                        help="The type of the model to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_mcdropout",
                        action='store_true',
                        help="Whether to run MC Dropout for uncertainty estimation.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--logfile_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the training log")
    parser.add_argument('--num_mc_sampling',
                        type=int,
                        default=50,
                        help="the number of MC sampling times when MC dropout")
    if bool(".txt" in sys.argv[1]) or bool(".json" in sys.argv[1]):
        config_file = sys.argv[1]
        with open(config_file) as file:
            lines = file.readlines()
        arguments = [l.strip() for l in lines]
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()
    return args

# load from arguments.txt file
def default_arguments(arguments):
    # para arguments given by a list in order as below args.
    args = Namespace(
        data_dir=arguments[0],
        weight_path_or_name=arguments[1],
        task_name=arguments[2],
        model_type=arguments[3],
        output_dir=arguments[4],
        cache_dir=arguments[5],
        max_seq_length=int(arguments[6]),
        do_train=True if arguments[7] == "True" else False,
        do_eval=True if arguments[8] == "True" else False,
        do_lower_case=True if arguments[9] == "True" else False,
        train_batch_size=int(arguments[10]),
        eval_batch_size=int(arguments[11]),
        learning_rate=float(arguments[12]),
        num_train_epochs=float(arguments[13]),
        warmup_proportion=float(arguments[14]),
        no_cuda=True if arguments[15] == "True" else False,
        overwrite_output_dir=True if arguments[16] == "True" else False,
        local_rank=int(arguments[17]),
        seed=int(arguments[18]),
        gradient_accumulation_steps=int(arguments[19]),
        fp16=True if arguments[20] == "True" else False,
        loss_scale=float(arguments[21]),
        server_ip=arguments[22],
        server_port=arguments[23],
        logfile_dir=arguments[24],
        device=arguments[25],
        n_gpu=int(arguments[26]),
    )
    return args