import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from dataclasses import dataclass
from typing import Optional
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertPooler
from transformers.file_utils import ModelOutput

from .. import config
from .loss import FocalLoss, PtrFocalLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class MyBertOutPut(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_cls: torch.FloatTensor = None
    logits_loc: torch.FloatTensor = None
    logits_tgt: torch.FloatTensor = None

class MyBertModel(BertPreTrainedModel):

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_config = model_config 
        self.task = model_config.task
        self.target = model_config.target
        self.bert = BertModel(model_config, add_pooling_layer=False)

        self.cls, self.loc, self.tgt = None, None, None
        if 'cls' in self.target:
            self.cls_bert_pooler = BertPooler(model_config)
            self.cls = nn.Linear(model_config.hidden_size, 2)
        if 'loc' in self.target:
            self.loc = nn.Linear(model_config.hidden_size, 1)
        if 'tgt' in self.target:
            if self.task == 'wrong-binary-operator':
                self.tgt_bert_pooler = BertPooler(model_config)
                self.tgt = nn.Linear(model_config.hidden_size, config.NUM_BIN_OPS)
            elif self.task in ('var-misuse', 'argument-swap'):
                self.tgt = nn.Linear(model_config.hidden_size, 1)
            else:
                assert(False)

        if not hasattr(self.model_config, 'pred_head') or self.model_config.pred_head == 'flat':
            self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        else:
            if 'cls' in self.target: self.cls_dropout = nn.Dropout(model_config.hidden_dropout_prob)
            if 'loc' in self.target: self.loc_dropout = nn.Dropout(model_config.hidden_dropout_prob)
            if 'tgt' in self.target: self.tgt_dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward_bert(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        cls_output, loc_output, tgt_output = None, None, None
        if not hasattr(self.model_config, 'pred_head') or self.model_config.pred_head == 'flat':
            seq_output = outputs.last_hidden_state
            seq_output = self.dropout(seq_output)
            if 'cls' in self.target:
                cls_output = self.cls_bert_pooler(seq_output)
            if 'loc' in self.target:
                loc_output = seq_output
            if 'tgt' in self.target:
                if self.task == 'wrong-binary-operator':
                    tgt_output = self.tgt_bert_pooler(seq_output)
                elif self.task in ('var-misuse', 'argument-swap'):
                    tgt_output = seq_output
                else:
                    assert(False)
        else:
            hidden_states = outputs.hidden_states
            targets = []
            for target in self.model_config.pred_head.split('-'):
                if target in self.target:
                    targets.append(target)
                elif target == 'locpos' and 'loc' in self.target:
                    targets.append('loc')
            for i, target in enumerate(targets):
                hidden_state = hidden_states[-1 * (len(targets)-i)]
                if target == 'cls':
                    cls_output = self.cls_dropout(hidden_state)
                    cls_output = self.cls_bert_pooler(cls_output)
                elif target in ('loc', 'locpos'):
                    loc_output = self.loc_dropout(hidden_state)
                elif target == 'tgt':
                    tgt_output = self.tgt_dropout(hidden_state)
                    if self.task == 'wrong-binary-operator':
                        tgt_output = self.tgt_bert_pooler(tgt_output)
                    elif self.task in ('var-misuse', 'argument-swap'):
                        tgt_output = tgt_output
                    else:
                        assert(False)
                else:
                    assert(False)

        return cls_output, loc_output, tgt_output

    def forward_cls(self, cls_output, cls_labels):
        if self.cls is None:
            return None, None

        loss_cls = None
        logits = self.cls(cls_output)
        if cls_labels is not None:
            mask = cls_labels != -1
            logits_valid, cls_labels_valid = logits[mask], cls_labels[mask]
            if logits_valid.size()[0] == 0:
                loss_cls = None
            else:
                if not hasattr(self.model_config, 'cls_loss') or self.model_config.cls_loss == 'ce':
                    cls_loss_func = CrossEntropyLoss(weight=None)
                elif self.model_config.cls_loss == 'focal':
                    cls_loss_func = FocalLoss(gamma=2, alpha=None)
                else:
                    assert(False)
                loss_cls = cls_loss_func(logits_valid, cls_labels_valid)

        return loss_cls, logits

    def forward_loc(self, loc_output, loc_candidate_masks, loc_corret_masks):
        if self.loc is None:
            return None, None

        logits_loc = self.loc(loc_output).squeeze(2)
        logits_loc = F.normalize(logits_loc)
        logits_loc = logits_loc.masked_fill(loc_candidate_masks == 0, 0 if self.training else torch.finfo(logits_loc.dtype).min)

        loss_loc = None
        if loc_corret_masks is not None:
            if not hasattr(self.model_config, 'ptr_loss') or self.model_config.ptr_loss == 'original':
                loss_func = PtrFocalLoss(gamma=0)
            elif self.model_config.ptr_loss == 'focal':
                loss_func = PtrFocalLoss(gamma=2)
            else:
                assert(False)
            loss_loc = loss_func(logits_loc, loc_corret_masks)

        return loss_loc, logits_loc

    def forward_tgt(self, logits_loc, tgt_output, loc_correct_masks, tgt_candidate_masks, tgt_correct_masks):
        if self.tgt is None:
            return None, None

        if self.task == 'wrong-binary-operator':
            logits_tgt = self.tgt(tgt_output)
        elif self.task in ('var-misuse', 'argument-swap'):
            logits_tgt = self.tgt(tgt_output).squeeze(2)
        else:
            assert(False)

        def get_masks_from_locs(_mask_dict, _locs):
            masks = []
            for i, loc in enumerate(_locs):
                if loc == -1 or loc == 0:
                    if self.task == 'wrong-binary-operator':
                        masks.append(torch.LongTensor([0] * config.NUM_BIN_OPS))
                    elif self.task in ('var-misuse', 'argument-swap'):
                        masks.append(torch.LongTensor([0] * config.MAX_SEQ_LEN))
                    else:
                        assert(False)
                else:
                    if torch.is_tensor(loc):
                        loc = int(loc)
                    if loc in _mask_dict[i]:
                        masks.append(torch.LongTensor(_mask_dict[i][loc]))
                    else:
                        masks.append(torch.LongTensor(_mask_dict[i][str(loc)]))
            masks = torch.stack(masks).to(device)
            return masks

        if self.training:
            correct_locs = []
            for mask in loc_correct_masks:
                for i in range(len(mask)):
                    if mask[i]:
                        correct_locs.append(i)
                        break
                else:
                    correct_locs.append(-1)
            tgt_candidate_masks = get_masks_from_locs(tgt_candidate_masks, correct_locs)
        else:
            locs = torch.argmax(logits_loc, dim=1)
            tgt_candidate_masks = get_masks_from_locs(tgt_candidate_masks, locs)

        logits_tgt = F.normalize(logits_tgt)
        logits_tgt = logits_tgt.masked_fill(tgt_candidate_masks == 0, 0 if self.training else torch.finfo(logits_tgt.dtype).min)

        loss_tgt = None
        if tgt_correct_masks is not None:
            if not hasattr(self.model_config, 'ptr_loss') or self.model_config.ptr_loss == 'original':
                loss_func = PtrFocalLoss(gamma=0)
            elif self.model_config.ptr_loss == 'focal':
                loss_func = PtrFocalLoss(gamma=self.model_config.ptr_loss_gamma)
            else:
                assert(False)
            loss_tgt = loss_func(logits_tgt, tgt_correct_masks)

        return loss_tgt, logits_tgt

    def forward_contrastive(self,
            input_ids1=None, attention_mask1=None, token_type_ids1=None,
            cls_labels1=None, loc_candidate_masks1=None, tgt_candidate_masks1=None, loc_correct_masks1=None, tgt_correct_masks1=None,
            input_ids2=None, attention_mask2=None, token_type_ids2=None,
            cls_labels2=None, loc_candidate_masks2=None, tgt_candidate_masks2=None, loc_correct_masks2=None, tgt_correct_masks2=None):
        cls_output1, loc_output1, tgt_output1 = self.forward_bert(input_ids1, attention_mask1, token_type_ids1)
        cls_output2, loc_output2, tgt_output2 = self.forward_bert(input_ids2, attention_mask2, token_type_ids2)

        loss_func = CosineEmbeddingLoss()
        loss = loss_func(cls_output1, cls_output2, torch.FloatTensor([-1] * cls_output1.size(0)).to(device)) 
        loss *= self.model_config.contrastive_weight

        loss_cls1, logits_cls1 = self.forward_cls(cls_output1, cls_labels1)
        loss_cls2, logits_cls2 = self.forward_cls(cls_output2, cls_labels2)
        if loss_cls1 is not None:
            loss += loss_cls1
        if loss_cls2 is not None:
            loss += loss_cls2

        loss_loc1, logits_loc1 = self.forward_loc(loc_output1, loc_candidate_masks1, loc_correct_masks1)
        loss_loc2, logits_loc2 = self.forward_loc(loc_output2, loc_candidate_masks2, loc_correct_masks2)
        if loss_loc1 is not None:
            loss += loss_loc1
        if loss_loc2 is not None:
            loss += loss_loc2

        loss_tgt1, logits_tgt1 = self.forward_tgt(logits_loc1, tgt_output1, loc_correct_masks1, tgt_candidate_masks1, tgt_correct_masks1)
        loss_tgt2, logits_tgt2 = self.forward_tgt(logits_loc2, tgt_output2, loc_correct_masks2, tgt_candidate_masks2, tgt_correct_masks2)
        if loss_tgt1 is not None:
            loss += loss_tgt1
        if loss_tgt2 is not None:
            loss += loss_tgt2

        return loss

    def forward(self,
            input_ids=None, attention_mask=None, token_type_ids=None,
            cls_labels=None, loc_candidate_masks=None, tgt_candidate_masks=None, loc_correct_masks=None, tgt_correct_masks=None,
            input_ids1=None, attention_mask1=None, token_type_ids1=None,
            cls_labels1=None, loc_candidate_masks1=None, tgt_candidate_masks1=None, loc_correct_masks1=None, tgt_correct_masks1=None,
            input_ids2=None, attention_mask2=None, token_type_ids2=None,
            cls_labels2=None, loc_candidate_masks2=None, tgt_candidate_masks2=None, loc_correct_masks2=None, tgt_correct_masks2=None, **kwargs):

        if 'contrastive' in self.target:
            loss = self.forward_contrastive(
                input_ids1, attention_mask1, token_type_ids1,
                cls_labels1, loc_candidate_masks1, tgt_candidate_masks1, loc_correct_masks1, tgt_correct_masks1,
                input_ids2, attention_mask2, token_type_ids2,
                cls_labels2, loc_candidate_masks2, tgt_candidate_masks2, loc_correct_masks2, tgt_correct_masks2,
            )

            return MyBertOutPut(loss, None, None, None)
        else:
            cls_output, loc_output, tgt_output = self.forward_bert(input_ids, attention_mask, token_type_ids)
            loss_cls, logits_cls = self.forward_cls(cls_output, cls_labels)
            loss_loc, logits_loc = self.forward_loc(loc_output, loc_candidate_masks, loc_correct_masks)
            loss_tgt, logits_tgt = self.forward_tgt(logits_loc, tgt_output, loc_correct_masks, tgt_candidate_masks, tgt_correct_masks)

            losses = []
            if loss_cls is not None:
                losses.append(loss_cls)
            if loss_loc is not None:
                losses.append(loss_loc)
            if loss_tgt is not None:
                losses.append(loss_tgt)

            loss = None
            if len(losses) > 0:
                loss = 0
                for l in losses:
                    loss += l

            return MyBertOutPut(loss, logits_cls, logits_loc, logits_tgt)
