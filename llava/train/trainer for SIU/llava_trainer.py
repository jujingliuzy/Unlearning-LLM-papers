import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Sampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from transformers import AutoProcessor
from llava.model import *
# if is_apex_available():
#     from apex import amp
def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
    outputs = model(**inputs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    loss /= gradient_accumulation_steps
    model.backward(loss)
    return loss
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

class LLaVATrainer(Trainer):
    def __init__(self, second_model_path,phrase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.second_model = LlavaLlamaForCausalLM.from_pretrained(
        #     second_model_path,
        #     torch_dtype=torch.float16,
        # )
        self.second_model = second_model_path
        self.phrase=phrase
        self.second_model, _, _, _ = deepspeed.initialize(model=self.second_model,
                                                          model_parameters=self.second_model.parameters(),
                                                          config='/data1/LLaVA/scripts/zero3.json')
        # print(self.second_model)
        # self.second_model.eval()
        # self.second_model = copy.deepcopy(self.model)


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            if self.args.group_by_modality_length:
                lengths = self.train_dataset.modality_lengths
                return LengthGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                    lengths=lengths,
                    group_by_modality=True,
                )
            else:
                return super()._get_train_sampler()





    def compute_kl_loss(self, batch, device):

        # print(model)
        # print(batch)
        temp = batch["input_ids"]<0
        batch["input_ids"][temp] = 0
        # print("batch--------------------")
        # print(batch['input_ids'][temp])
        #batch["input_ids"][temp] = torch.tensor(batch["input_ids"][temp],)
        # print(batch["input_ids"].shape)
        # normal_outputs = model(
        #     batch["input_ids"].to(device),
        #     attention_mask=batch["attention_mask"].int().to(device),
        #     labels=batch["labels"].to(device),
        # )

        # self.second_model.to(device)
        # with torch.no_grad():
        #     pretrained_outputs = self.second_model(
        #         batch["input_ids"].to(device),
        #         attention_mask=batch["attention_mask"].int().to(device),
        #     )
        # mask = batch["mask"]  # 获取掩码
        # mask_unsqueezed = mask.unsqueeze(-1)  # 增加一个新维度，形状变为 [16, 92, 1]
        # mask = mask_unsqueezed.expand(-1, -1, 32000)
        # print(pretrained_outputs.logits.shape,"shape")
        # prob_p = F.softmax(outputs.logits, dim=-1)
        # prob_q = F.softmax(normal_outputs.logits, dim=-1)
        kl_loss = 0
        # if mask is not None:
        #     mask_inverse = 1 - mask
        #     kl_loss = -((prob_p * torch.log(prob_q + 1e-12)) * mask_inverse).sum() / mask_inverse.sum()

        # 计算交叉熵损失
        # target_loss = 0
        # if mask is not None:
        #     logits_flat = normal_outputs.logits.view(-1, normal_outputs.logits.size(-1))
        #     labels_flat = batch["labels"].view(-1)
        #     cross_entropy_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none').view_as(batch["labels"])
        #     target_loss = (cross_entropy_loss * mask).sum() / mask.sum()

        return prob_p

    # def training_step(self, model, inputs):
    #         model.train()
    #         device = next(model.parameters()).device

    #         kl_loss, another_loss = self.compute_kl_loss(model, inputs, device)

    #         # Assuming another_loss calculation here
    #         # another_loss = torch.tensor(1.0, device=device)

    #         # combined_loss = 0.1*kl_loss + 0.9*another_loss
    #         combined_loss = 0.9*another_loss

    #         if self.args.n_gpu > 1:
    #             combined_loss = combined_loss.mean()

    #         self.accelerator.backward(combined_loss)

    #         return combined_loss.detach()

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     device = next(model.parameters()).device

    #     kl_loss, another_loss = self.compute_kl_loss(model, inputs, device)

    #     # Assuming another_loss calculation here
    #     # another_loss = torch.tensor(1.0, device=device)

    #     # combined_loss = 0.1*kl_loss + 0.9*another_loss
    #     combined_loss = 0.9*another_loss
    #     del inputs['mask']
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)

    #     return loss.detach() / self.args.gradient_accumulation_steps


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        self.model.train()

        input2=  dict(
            input_ids=inputs['input_ids'].detach(),
            attention_mask=inputs['attention_mask'].detach()
        )
        inputs = self._prepare_inputs(inputs)
        device = next(model.parameters()).device
        tokens = self.tokenizer.tokenize(self.phrase)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # kl_loss, another_loss = self.compute_kl_loss(input2, device)
        # pro_p = self.compute_kl_loss(input2, device).detach()
        # print(f"Tokens: {tokens}")
        # print(f"Token IDs: {token_ids}")
        mask_phr = torch.ones([32000], dtype=torch.float32).to(self.args.device)
        for id in token_ids:
            mask_phr[id]=0
        mask=inputs['mask']
        del inputs['mask']
        with torch.no_grad():
            outputs = self.second_model(**inputs)
        # print(inputs['input_ids'].shape,"input")
        # print(inputs['labels'].shape,"labels")
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss,normal_outputs = self.compute_loss(model, inputs,return_outputs=True)
        # print(normal_outputs.logits.shape,"normal")
        # print(outputs.logits.shape,"out")
        mask_unsqueezed = mask.unsqueeze(-1)  # 增加一个新维度，形状变为 [16, 92, 1]
        mask = mask_unsqueezed.expand(-1, -1, 32000)
        prob_p = F.softmax(outputs.logits, dim=-1)*mask_phr
        prob_q = F.softmax(normal_outputs.logits, dim=-1)*mask_phr
        prob_p=prob_p[:,:mask.size(1),:]
        prob_q=prob_q[:,:mask.size(1),:]
        # print(prob_p.shape, "pobpshape")
        if mask is not None:
            mask_inverse = 1 - mask
            # print(mask.sum(),"masksum")
            # print(mask_inverse.shape, "maskshape")
            # print(prob_p.shape, "ppshape")
            # print(prob_q.shape, "proqshape")

            kl_loss = -((prob_p * torch.log(prob_q + 1e-12)) * mask_inverse).sum() / mask_inverse.sum()
        # pretrained_outputs = self.second_model(
        #     inputs["input_ids"].to(device),
        #     attention_mask=inputs["attention_mask"].int().to(device),
        #     labels=inputs["labels"].to(device),
        # )
        loss = loss + kl_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps





    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
