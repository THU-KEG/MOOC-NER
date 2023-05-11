# -*- coding:utf-8 -*
import logging
import os
import json
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
 
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,field_embeds):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.field_embeds = field_embeds

def read_examples_from_file(args, data_dir, mode):
    
    # if mode == "train":
    #     mode = str(args.noise_ratio)+"-"+mode
    file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        
        for item in data:
            words = item["str_words"]
            labels = item["tags"]
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels))
            guid_index += 1
    
    return examples


def read_field_embeddings_from_file(file_path):
    data = pd.read_csv(file_path)# make sure csv file has the header
    X = data.iloc[:,0].values.tolist()
    Y = data.iloc[:,1:].values.tolist()
    
    # build map:
    field_embeddings_map = {}
    
    assert(len(X)==len(Y))
    length = len(X)
    for i in range(length):
        field_embeddings_map[X[i]]=Y[i]
    
    logger.info('finish read %d field embeddings from %s',length,file_path)
    return field_embeddings_map  



def convert_examples_to_features(
    examples,
    field_embeddings_map,
    field_embeds_size,
    max_seq_length,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = 1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = [] 
    extra_long_samples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        field_embeds = []

        word_index = 0
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if(len(word_tokens) == 0):
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))

            if label == 0:
                # not taged:
                field_embeds.extend([[pad_token]*field_embeds_size]*len(word_tokens))
            elif label == 1:
                # find the end_index
                for word_ind in range(word_index,len(example.words)):
                    if example.labels[word_ind]==0:
                        end_index =word_ind
                        break
                # get the concept:
                concept = ''.join(example.words[word_index:end_index])
                # get the embedding:
                if concept not in field_embeddings_map.keys():
                    field_embeds.extend([[pad_token]*field_embeds_size]*len(word_tokens))
                else:
                    field_embeds.extend([field_embeddings_map[concept]]*len(word_tokens))
            else:
                #find the begin_index and end_index:
                for word_ind in range(1,word_index):
                    if example.labels[word_index - word_ind]==1:
                        begin_index = word_index - word_ind
                        break
                for word_ind in range(word_index,len(example.words)):
                    if example.labels[word_ind] == 0:
                        end_index = word_ind
                        break   
                # get the concept:
                concept = ''.join(example.words[begin_index:end_index])
                # get the embedding:
                if concept not in field_embeddings_map.keys():
                    field_embeds.extend([[pad_token]*field_embeds_size]*len(word_tokens))
                else:
                    field_embeds.extend([field_embeddings_map[concept]]*len(word_tokens))
            word_index += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            field_embeds = field_embeds[:(max_seq_length-special_tokens_count)]
            extra_long_samples += 1


        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        field_embeds += [[pad_token] *  field_embeds_size]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            field_embeds += [[pad_token] *  field_embeds_size]        
        segment_ids = [sequence_a_segment_id] * len(tokens)

        
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        field_embeds += [[pad_token] *  field_embeds_size]        

        segment_ids = [cls_token_segment_id] + segment_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # pad on the right
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length  
        field_embeds += [[pad_token]*field_embeds_size]* padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(field_embeds) == max_seq_length
        
        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info('field_embeds : %s',' '.join([str(x) for x in field_embeds]))
        
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids,field_embeds=field_embeds)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    return features


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "{}_{}.pt".format(
            args.dataset, mode
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args, args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            read_field_embeddings_from_file(args.embeds_file_path),
            args.fields_size,
            args.max_seq_length,
            tokenizer,
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0 ,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0 ,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_field_embeds = torch.tensor([f.field_embeds for f in features],dtype = torch.float)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_field_embeds, all_ids)
    return dataset

def get_labels(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        labels = []
        with open(path+dataset+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            for l, _ in data.items():
                labels.append(l)
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-Concept", "I-Concept"]

def tag_to_id(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data
    else:
        return {"O": 0, "B-Concept": 1, "I-Concept": 2}
        
if __name__ == '__main__':
    save(args)