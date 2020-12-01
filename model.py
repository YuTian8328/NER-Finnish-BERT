import config
import torch
import transformers
import torch.nn as nn

def loss_fn(output,target,mask,num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits,active_labels)
    return loss
class EntityModel(nn.Module):
    def __init__(self,num_tag):
        super(EntityModel,self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.bert_drop_1=nn.Dropout(0.3)
        # self.bert_drop_2=nn.Dropout(0.3)
        self.num_tag = num_tag
        self.out_tag = nn.Linear(768, self.num_tag)
        # self.out_pos = nn.Linear(768, self.num_pos)
    def forward(self, ids, mask, token_type_ids, target_tag):
        ol,_ = self.bert(ids, attention_mask=mask,token_type_ids=token_type_ids)
        bo_tag = self.bert_drop_1(ol)
        # bo_pos = self.bert_drop_2(ol)

        tag = self.out_tag(bo_tag)
        # pos = self.out_pos(bo_pos)
 
        loss = loss_fn(tag, target_tag, mask, self.num_tag)
        # loss_pos= loss_fn(pos, target_pos,self.num_pos)

        # loss = (loss_tag + loss_pos)/2
        return tag, loss
