# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from autorag.utils.util import empty_cuda_cache


class EncT5ForSequenceClassification(T5PreTrainedModel):
	_keys_to_ignore_on_load_missing = [
		r"encoder\.embed_tokens\.weight",
	]

	def __init__(self, config: T5Config, dropout=0.1):
		super().__init__(config)
		try:
			from torch import nn
		except ImportError:
			raise ImportError("Please install PyTorch to use TART reranker.")
		self.num_labels = config.num_labels
		self.config = config

		self.shared = nn.Embedding(config.vocab_size, config.d_model)

		encoder_config = copy.deepcopy(config)
		encoder_config.use_cache = False
		encoder_config.is_encoder_decoder = False
		self.encoder = T5Stack(encoder_config, self.shared)

		self.dropout = nn.Dropout(dropout)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()

		# Model parallel
		self.model_parallel = False
		self.device_map = None

	def parallelize(self, device_map=None):
		try:
			import torch
		except ImportError:
			raise ImportError("Please install PyTorch to use TART reranker.")
		self.device_map = (
			get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
			if device_map is None
			else device_map
		)
		assert_device_map(self.device_map, len(self.encoder.block))
		self.encoder.parallelize(self.device_map)
		self.classifier = self.classifier.to(self.encoder.first_device)
		self.model_parallel = True

	def deparallelize(self):
		self.encoder.deparallelize()
		self.encoder = self.encoder.to("cpu")
		self.model_parallel = False
		self.device_map = None
		empty_cuda_cache()

	def get_input_embeddings(self):
		return self.shared

	def set_input_embeddings(self, new_embeddings):
		self.shared = new_embeddings
		self.encoder.set_input_embeddings(new_embeddings)

	def get_encoder(self):
		return self.encoder

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		try:
			import torch
			from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
		except ImportError:
			raise ImportError("Please install PyTorch to use TART reranker.")
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			inputs_embeds=inputs_embeds,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		loss = None
		if labels is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (
					labels.dtype == torch.long or labels.dtype == torch.int
				):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels)
		if not return_dict:
			output = (logits,) + outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
