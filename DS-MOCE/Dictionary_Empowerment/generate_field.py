"""
generate field 
"""

from ast import Num
from email.policy import default
from lib2to3.pgen2 import token
import os
import torch
import torch.nn.functional as F
from arguments import get_args
from pretrain_glm import initialize_distributed
from pretrain_glm import set_random_seed
from pretrain_glm import get_masks_and_position_ids
from utils import load_checkpoint
from configure_data import prepare_tokenizer
from generation_utils import BeamSearchScorer

from torch.utils import data
import numpy as np
import pandas as pd
import csv
import time

from train_utils import get_model
from sklearn.metrics import average_precision_score

np.seterr(divide='ignore',invalid='ignore')


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args, model_type="generation")

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load_pretrained is not None:
        args.no_load_optim = True
        args.load = args.load_pretrained
        _ = load_checkpoint(
            model, None, None, args)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, args,device):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.

    attention_mask = torch.tensor([tokens.size(1)], device=device, dtype=torch.long)
    position_ids = torch.arange(tokens.size(1), device=device, dtype=torch.long)
    if not args.no_block_position:
        block_position_ids = torch.zeros(tokens.size(1), device=device, dtype=torch.long)
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids



# define:
all_field = [
    '心理学', '教育学', '语言学',
    '世界历史', '数学', '物理学', '化学','机械工程',
    '材料科学技术', '电气工程',  '建筑学', 
    '航空科学技术', '医学', '管理科学技术', '计算机科学技术','农学',
    '力学','船舶工程','航天科学技术','免疫学',
]

# define:
field_short = [
    '心理', '教育', '语言',
    '历史', '数学', '物理', '化学',
    '机械',
    '材料',  '电气',  '建筑', 
    '航空', '医学', '管理', '计算机','农业',
    '力学','船舶','航天','免疫'
]

def generate_prompt(prompt, concept, tokenizer):
    """
    generate sentence according to prompt, return decode ids
    e.g. 
    prompt framework(string): "今天我们继续来讲解[MASK]领域课程，我们的主题是[concept]。"
    concept(string) : "傅立叶变换"
    return : sentence_tokens_tensor, sentence_length
    """

    raw_text = prompt.replace("[concept]",concept)
    sentence_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
    
    # block LM so we add 'CLS'/'ENC' at the beginning
    # and 'eos' at the end
    sentence_tokens =[tokenizer.get_command('ENC').Id] + sentence_tokens + [tokenizer.get_command('eos').Id]

    sentence_length = len(sentence_tokens)

    # tensor
    sentence_tokens_tensor = torch.cuda.LongTensor(sentence_tokens)

    return sentence_tokens_tensor,sentence_length

def get_field_score(model, tokenizer, sentence_tokens, position, field_list,args):
    """
    zero-shot [MASK] cloze-filling results
    return field score
    that means [MASK] position tokens : list[sop, fieldlist, eop]
    """
    
    mems = []
    tokens,attention_mask, position_ids = get_batch(sentence_tokens,args, torch.cuda.current_device())
    _, *mems = model(tokens, position_ids,attention_mask,*mems)
   
        
    # tokens are [MASK] cloze-filling results.
    # beginning with <start of piece> and iteratively predict next tokens 
    tokens = sentence_tokens.new_full((1,1),tokenizer.get_command('sop').Id)
    
    counter = 0

    # iterate to generate predict_list and get scores
    predict_list = field_list + [tokenizer.get_command('eop').Id] 
    scores = 0

    while counter < len(predict_list):
        # print("counter #",counter)
        position_ids = sentence_tokens.new_ones(1,2,1)
        position_ids[:,0]=position
        position_ids[:,1]=counter + 1

        attention_mask = sentence_tokens.new_zeros([1],device = sentence_tokens.device, dtype = torch.long)

        last_token = tokens[:,-1:]

        next_token_logits, *mems = model(last_token,position_ids,attention_mask, *mems)

        next_token_logits = next_token_logits[:,-1] # size (1, vocab_size)
        next_token_logits /= args.temperature
        log_probs = F.log_softmax(next_token_logits,dim = -1)
        

        # debug:
        # field_now = [tokenizer.DecodeIds(field_list)]
        # next_decode = tokenizer.IdToToken(predict_list[counter])
        # score_update = log_probs[0,predict_list[counter]].item() 
        # print("#field: {}, next_decode: {}, score_update {}".format(field_now,next_decode,score_update))

        scores += log_probs[0,predict_list[counter]].item()
        if args.short_setting:
            return scores # only the first token

        # update
        
        next_token = sentence_tokens.new_full((1,1),predict_list[counter])
        tokens = torch.cat((tokens, next_token), dim = 1) # bug here! dim = 1
        # print("update token: ",tokens)
        counter += 1 # bug here! counter last update
    
    return scores




def sample_scores(model, tokenizer, sentence_tokens, position,args):
    """
    get 20 shape embeddings
    """
    score_list = []

    if args.short_setting:
        # we use different field name here.
        # field_short and all_field
        for field in field_short:
            id = tokenizer.EncodeAsIds(field).tokenization[0]
            if id == 43358:
                id = tokenizer.EncodeAsIds(field).tokenization[1]
            score_list.append(get_field_score(model,tokenizer,sentence_tokens,position,[id],args))
    else:
        fields_ids = [tokenizer.EncodeAsIds(field).tokenization for field in all_field]

    # for i, per_field_list in enumerate(fields_ids):
    #     print("field # {} with ids {}.".format(all_field[i],per_field_list))
    #     for per_num in per_field_list:
    #         print("Decode back: ",tokenizer.IdToToken(per_num))

        for per_field_list in fields_ids:

            # debug: remove "_" with ID 43358
            if 43358 in per_field_list:
                # print("remove the _!")
                score_list.append(get_field_score(model,tokenizer,sentence_tokens,position, per_field_list[1:],args))
            else:
                score_list.append(get_field_score(model,tokenizer,sentence_tokens,position, per_field_list,args))

    
    # debug now
    # print(list(sorted(zip(all_field,score_list),key= lambda x:-x[1])))

    return score_list



class Concept_Dataset(data.Dataset):
    def __init__(self,path,tokenizer,prompt) -> None:
        super().__init__()

        # read file
        data = pd.read_csv(path) # make sure csv file has the header

        
        self.X = data.iloc[:,0].values.tolist()
        self.Y = data.iloc[:,1:].values.tolist()

        # turn label into one-hot
        # according to all_field 
        self.one_hot_Y = np.zeros((len(self.X),20))
        for i,label_list in enumerate(self.Y):
            positions = [pos for pos,label in enumerate(all_field) if label in label_list]
            for position in positions:
                self.one_hot_Y[i,position]=1

        self.tokenizer = tokenizer
        self.prompt = prompt
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """
        generate sentence according to prompt, return decode ids
        e.g. 
        prompt framework(string): "今天我们继续来讲解[MASK]领域课程，我们的主题是[concept]。"
        concept(string) : "傅立叶变换"
        return : sentence_tokens_tensor, sentence_length
        """
        raw_text = self.prompt.replace("[concept]",self.X[index])
        sentence_tokens = self.tokenizer.EncodeAsIds(raw_text).tokenization
        
        # block LM so we add 'CLS'/'ENC' at the beginning
        # and 'eos' at the end
        sentence_tokens =[self.tokenizer.get_command('ENC').Id] + sentence_tokens + [self.tokenizer.get_command('eos').Id]

        position = sentence_tokens.index(self.tokenizer.get_command('MASK').Id)
        sentence_length = len(sentence_tokens)

        # tensor
        sentence_tokens_tensor = torch.cuda.LongTensor(sentence_tokens)

        return {"concept": self.X[index], "text":raw_text, "X_tensor": sentence_tokens_tensor, "length" :sentence_length, "labels":self.Y[index], "position":position}

    def write(self,score_lists,output_path,skip_header=False):
        """
        write the 20 embeddings into a csv file 
        also output the metrics
        """

        print("generate 20 dim embeddings for each concept.")

        with open(output_path,'w') as csvfile:
            writer = csv.writer(csvfile)
            if not skip_header:
                header = ["概念"] + all_field
                writer.writerow(header)
            for i,row in enumerate(score_lists):
                # print(self.X[i])
                # print(row)
                rows = [self.X[i]]+ row.tolist()
                writer.writerow(rows)
        
        print("evaluate the results")

        #debug: softmax:score_lists
        y_true = np.array(self.one_hot_Y)
        y_score = np.array(score_lists)

        num = self.__len__()
        AP = 0.
        for i in range(num):
            AP += average_precision_score(y_true[i],y_score[i])
        
        print("mAP : ",AP/num)

def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # prepare data:
    data_set =Concept_Dataset(
        args.concept_dictionary,
        tokenizer,
        prompt=args.prompt_template
    )
    
    args.batch_size = 1

    data_loader = data.DataLoader(data_set,args.batch_size)

    # Model, optimizer, and learning rate.
    model = setup_model(args)
    # setting default batch size to 1

    predict_labels = np.zeros((data_loader.__len__(),20))

    start_time = time.time()
    for i,per_concept_data in enumerate(data_loader):
        if i % 100 == 0:
            print("\nTaken time {:.2f}\n".format(time.time() - start_time))
            print("done {}/{}".format(i,data_set.__len__()))
            start_time = time.time()

        sentence_tokens_tensor = per_concept_data['X_tensor']
        # print("DEBUG: raw_concept",per_concept_data['concept'])
        this_concept_score = sample_scores(model,tokenizer,sentence_tokens_tensor,per_concept_data['position'],args)
        this_concept_score = torch.tensor(this_concept_score)
        predict_labels[i,:]=F.softmax(this_concept_score,dim = -1 )

    # print("DEBUG: predict_labels: ",predict_labels)
    data_set.write(predict_labels,
        args.concept_output_embeddings
    )

if __name__ == '__main__':
    main()