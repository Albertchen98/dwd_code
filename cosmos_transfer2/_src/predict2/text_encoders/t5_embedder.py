import abc
import functools
from enum import Enum
from typing import Any, ClassVar, Literal, TypeAlias, overload

import attrs
import torch
import transformers
from transformers import T5EncoderModel, T5TokenizerFast
from typing_extensions import Self, assert_never, override

from cosmos_transfer2._src.imaginaire.utils import log

transformers.utils.logging.set_verbosity_error()

NUM_EMBEDDING_PADDING_TOKENS = 512

class CosmosTextEncoderBase(torch.nn.Module, abc.ABC):
    @overload
    def encode_prompts(self, prompts: str, max_length: int, return_mask: Literal[False]) -> torch.Tensor: ...
    @overload
    def encode_prompts(
        self, prompts: list[str], max_length: int, return_mask: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @abc.abstractmethod
    def encode_prompts(
        self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into hidden state representations.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to num_tokens.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.
        """

@attrs.define(slots=False)
class CosmosT5TextEncoderConfig:
    """
    Config for the T5 text encoder model
    """

    CKPT_PATH: ClassVar[str] = "/cache/checkpoints/google-t5/t5-11b"
    NUM_TOKENS: ClassVar[int] = 512
    EMBED_DIM: ClassVar[int] = 1024

    ckpt_path: str = CKPT_PATH
    num_tokens: int = NUM_TOKENS
    embed_dim: int = EMBED_DIM
    compute_online: bool = True


class CosmosT5TextEncoder(CosmosTextEncoderBase):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        config: CosmosT5TextEncoderConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = torch.bfloat16,
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: The device to use for computations.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.tokenizer = T5TokenizerFast.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype)
        self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
        self.text_encoder.eval()

        log.info("T5 Text encoder model instantiated")

    @property
    def model(self) -> Self:
        return self

    @torch.inference_mode()
    def compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor], input_caption_key: str,max_length: int | None = None, return_mask: bool = False):
        prompts = data_batch[input_caption_key]
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")
        if max_length is None:
            max_length = self.config.num_tokens

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        if return_mask:
            return encoded_text, attn_mask.bool()
        return encoded_text
    
    @override
    @torch.inference_mode()
    def encode_prompts(self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")
        if max_length is None:
            max_length = self.config.num_tokens

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        if return_mask:
            return encoded_text, attn_mask.bool()
        return encoded_text