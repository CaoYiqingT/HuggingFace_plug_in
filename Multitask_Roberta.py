#%%
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import gelu

from dataclasses import dataclass
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.utils import logging
from transformers.modeling_outputs import ModelOutput

logger = logging.get_logger(__name__)

@dataclass
class MultitaskOutput(ModelOutput):
    """
    Base class for multitask outputs.

    Args:
        roberta_sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total loss for multiple tasks

        mlm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.

        mlm_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        sequence_classification_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.

        sequence_classification_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).

        multiple_choice_loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.

        multiple_choice_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above). Classification scores (before SoftMax).

        token_classification_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.

        token_classification_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).

        qa_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

        qa_start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).

        qa_end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    roberta_sequence_output: torch.FloatTensor = None

    loss: Optional[torch.FloatTensor] = None

    mlm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None

    sequence_classification_loss: Optional[torch.FloatTensor] = None
    sequence_classification_logits: torch.FloatTensor = None

    multiple_choice_loss: Optional[torch.FloatTensor] = None
    multiple_choice_logits: torch.FloatTensor = None

    token_classification_loss: Optional[torch.FloatTensor] = None
    token_classification_logits: torch.FloatTensor = None

    qa_loss: Optional[torch.FloatTensor] = None
    qa_start_logits: torch.FloatTensor = None
    qa_end_logits: torch.FloatTensor = None

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaForMultipleTask(RobertaPreTrainedModel):
    """
    Multitask Roberta Model, support the following tasks:
        - MLM
        - SequenceClassification
        - MultipleChoice
        - TokenClassification
        - QuestionAnswering
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *multitask, **task_specific_param):
        super().__init__(config)
        self.multitask = multitask
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if "MLM" in self.multitask:
            self.lm_head = RobertaLMHead(config)
            # The LM head weights require special treatment only when they are tied with the word embeddings
            self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        if "SequenceClassification" in self.multitask:
            self.sequence_classification_num_labels = task_specific_param["sequence_classification_num_labels"]
            self.sequence_classification_classifier = RobertaClassificationHead(config, self.sequence_classification_num_labels)
        if "MultipleChoice" in self.multitask:
            self.pooler = RobertaPooler(config)
            self.pooler_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.multiple_choice_classifier = nn.Linear(config.hidden_size, 1)
        if "TokenClassification" in self.multitask:
            self.token_classification_num_labels = task_specific_param["token_classification_num_labels"]
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.classifier_dropout = nn.Dropout(classifier_dropout)
            self.token_classification_classifier = nn.Linear(config.hidden_size, self.token_classification_num_labels)
        if "QuestionAnswering" in self.multitask:
            self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()       

    def get_output_embeddings(self):
        return self.lm_head.decoder if "MLM" in self.multitask else None

    def set_output_embeddings(self, new_embeddings):
        if "MLM" in self.multitask:
            self.lm_head.decoder = new_embeddings
        else:
            logger.warning("MLM not in training tasks, thus set_output_embeddings is not supported")

    def get_sequence_classification_loss(self, logits, labels):
        loss = None
        if labels is not None:
            if self.sequence_classification_num_labels == 1:
                self.sequence_classification_problem_type = "regression"
            elif self.sequence_classification_num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.sequence_classification_problem_type = "single_label_classification"
            else:
                self.sequence_classification_problem_type = "multi_label_classification"

            if self.sequence_classification_problem_type == "regression":
                loss_fct = MSELoss()
                if self.sequence_classification_num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.sequence_classification_problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.sequence_classification_num_labels), labels.view(-1))
            elif self.sequence_classification_problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss

    def get_question_answering_loss(self, start_logits, end_logits, start_positions, end_positions):
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        sequence_classification_labels: Optional[torch.LongTensor] = None,
        multiple_choice_labels: Optional[torch.LongTensor] = None,
        token_classification_labels: Optional[torch.LongTensor] = None,
        qa_start_positions: Optional[torch.LongTensor] = None,
        qa_end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultitaskOutput]:
        r"""
        - mlm_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        - sequence_classification_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        - multiple_choice_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        - token_classification_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., token_classification_num_labels - 1]`.
        - qa_start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        - qa_end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "MultipleChoice" in self.multitask:
            multiple_choice_num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

            input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
            inputs_embeds = (
                inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                if inputs_embeds is not None
                else None
            )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        masked_lm_loss = None
        sequence_classification_loss = None
        multiple_choice_loss = None
        token_classification_loss = None
        qa_loss = None

        if "MLM" in self.multitask:
            mlm_prediction_scores = self.lm_head(sequence_output)
            
            if mlm_labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(mlm_prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        if "SequenceClassification" in self.multitask:
            sequence_classification_logits = self.sequence_classification_classifier(sequence_output)

            if sequence_classification_labels is not None:
                sequence_classification_loss = self.get_sequence_classification_loss(sequence_classification_logits, sequence_classification_labels)
        if "MultipleChoice" in self.multitask:
            pooled_output = self.pooler(sequence_output)
            pooled_output = self.pooler_dropout(pooled_output)
            multiple_choice_logits = self.multiple_choice_classifier(pooled_output)
            multiple_choice_logits = multiple_choice_logits.view(-1, multiple_choice_num_choices)

            if multiple_choice_labels is not None:
                loss_fct = CrossEntropyLoss()
                multiple_choice_loss = loss_fct(multiple_choice_logits, multiple_choice_labels)
        if "TokenClassification" in self.multitask:
            droped_sequence_output = self.classifier_dropout(sequence_output)
            token_classification_logits = self.token_classification_classifier(droped_sequence_output)

            if token_classification_labels is not None:
                loss_fct = CrossEntropyLoss()
                token_classification_loss = loss_fct(token_classification_logits.view(-1, self.token_classification_num_labels), token_classification_labels.view(-1))
        if "QuestionAnswering" in self.multitask:
            qa_logits = self.qa_outputs(sequence_output)
            qa_start_logits, qa_end_logits = qa_logits.split(1, dim=-1)
            qa_start_logits = qa_start_logits.squeeze(-1).contiguous()
            qa_end_logits = qa_end_logits.squeeze(-1).contiguous()

            if qa_start_positions is not None and qa_end_positions is not None:
                qa_loss = self.get_question_answering_loss(qa_start_logits, qa_end_logits, qa_start_positions, qa_end_positions)
        
        multitask_loss = None
        for task_loss in (masked_lm_loss, sequence_classification_loss, multiple_choice_loss, token_classification_loss, qa_loss):
            if task_loss is not None:
                multitask_loss += task_loss

        # if not return_dict:
        #     output = (mlm_prediction_scores,) + outputs[2:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MultitaskOutput(
            roberta_sequence_output=sequence_output,
            loss=multitask_loss,
            mlm_loss=masked_lm_loss,
            mlm_logits=mlm_prediction_scores,
            sequence_classification_loss=sequence_classification_loss,
            sequence_classification_logits=sequence_classification_logits,
            multiple_choice_loss=multiple_choice_loss,
            multiple_choice_logits=multiple_choice_logits,
            token_classification_loss=token_classification_loss,
            token_classification_logits=token_classification_logits,
            qa_loss=qa_loss,
            qa_start_logits=qa_start_logits,
            qa_end_logits=qa_end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# %%
