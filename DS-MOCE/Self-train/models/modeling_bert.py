from transformers import BertModel,BertPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss

class BERTForTokenClassification_v2(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.embedding = nn.Linear(20, config.hidden_size)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    
    def forward(self, field_embeds,input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, label_mask=None):

        bert_embedding = self.get_input_embeddings() 
        # embedding object 
        # see more: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

        embeds_1 = bert_embedding(input_ids) 
        embeds_2 = self.embedding(field_embeds)
        inputs_embeds = torch.add(embeds_1,embeds_2)

        outputs = self.bert(input_ids=None,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,sequence_output) + outputs[2:]  # add hidden states and attention if they are here
        loss_dict = {}
        if labels is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # active_loss = True
                # if attention_mask is not None:
                #     active_loss = attention_mask.view(-1) == 1
                # if label_mask is not None:
                #     active_loss = active_loss & label_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
            
            for key in labels:
                label = labels[key]
                if label is None:
                    continue
                # if key=="pseudo" and label_mask is not None:
                if label_mask is not None:
                    all_active_loss = active_loss & label_mask.view(-1)
                else:
                    all_active_loss = active_loss
                active_logits = logits.view(-1, self.num_labels)[all_active_loss]

                if label.shape == logits.shape:
                    loss_fct = KLDivLoss()
                    # loss_fct = SoftFocalLoss(gamma=2)
                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1, self.num_labels)[all_active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits, label)
                else:
                    loss_fct = CrossEntropyLoss()
                    # loss_fct = FocalLoss(gamma=2)
                    # loss_fct = NLLLoss()
                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1)[all_active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
                loss_dict[key] = loss


            outputs = (loss_dict,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
