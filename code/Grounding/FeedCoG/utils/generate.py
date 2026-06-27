import torch

from transformers.utils import logging

from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

logger = logging.get_logger(__name__)

@torch.no_grad()
def generate(
    model: GenerationMixin,
    cfg_scale: float = None,
    enhance_query: bool = False,
    # === Sampling 参数 ===
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = None,
    **kwargs,
):
    """兼容性更强的生成函数：直接使用 model.generate，避免调用私有API。

    仅返回必要的 output_ids，满足下游 decode 需求。
    
    新增 sampling 支持：
    - do_sample: 是否启用采样 (True=sampling, False=greedy)
    - temperature: 采样温度 (默认 1.0)
    - top_p: nucleus sampling 阈值 (默认 0.9)
    - seed: 随机种子 (用于复现性)
    """
    # 提取控制参数
    max_new_tokens = kwargs.pop("max_new_tokens", None)
    return_scores = kwargs.pop("return_scores", False)  # 占位，不使用
    tokenizer = kwargs.pop("tokenizer", None)  # 未使用，仅保持接口

    # 保持输入张量不变，直接透传给 model.generate
    model_inputs = kwargs

    # 生成参数
    gen_kwargs = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    
    # Sampling 配置
    gen_kwargs["do_sample"] = do_sample
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        # 设置随机种子以保证可复现性
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    outputs = model.generate(**model_inputs, **gen_kwargs)

    # 兼容不同 Transformers 版本的返回
    output_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs

    return dict(output_ids=output_ids)

def sample(
    model: GenerationMixin,
    input_ids: torch.LongTensor,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    cfg_scale: float = None,
    enhance_query: bool = False,
    return_scores: bool = False,
    topk: int = 10,
    **model_kwargs,
):
    # init values
    if return_scores:
        assert input_ids.size(0) == 1, "Only batch size 1 is supported in return_scores mode."
        topk_scores, topk_indices = [], []
        value_scores, sub_value_scores = [], []
        value_indices = torch.arange(15, 25).tolist() # according to the Qwen Codebook
        log_prob_list = []
    # 兼容不同实现下的 pad token 获取
    pad_token_id = getattr(generation_config, "_pad_token_tensor", None)
    if pad_token_id is None:
        pad_id_int = getattr(generation_config, "pad_token_id", None)
        if pad_id_int is None:
            pad_id_int = 0
        pad_token_id = torch.tensor(pad_id_int, device=input_ids.device)
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    # 版本兼容：不同transformers实现的签名不一致
    try:
        # 常见签名：(input_ids, model_kwargs)
        model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    except TypeError:
        try:
            # 另一类签名：(model_kwargs)
            model_kwargs = model._get_initial_cache_position(model_kwargs)
        except TypeError:
            # 有的实现可能不需要显式初始化，保持原样
            model_kwargs = model_kwargs

    while model._has_unfinished_sequences(
        this_peer_finished, False, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        # NOTE: Clone is needed
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # token selection
        if cfg_scale is not None:
            next_token_logits_cond = next_token_logits[0:1]
            next_token_logits_uncond = next_token_logits[1:2]
            next_token_logits = (
                next_token_logits_uncond + cfg_scale * (
                    next_token_logits_cond - next_token_logits_uncond
                )
            )
            next_tokens = torch.argmax(next_token_logits, dim=-1).expand(2)
        elif enhance_query:
            num_enhance = next_token_logits.shape[0]
            next_token_logits = next_token_logits.mean(dim=0, keepdim=True)
            next_tokens = torch.argmax(next_token_logits, dim=-1).expand(num_enhance)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        if return_scores:
            if cfg_scale is not None:
                _ref = next_token_logits[0:1]
            else:
                _ref = next_token_logits
            cur_log_prob = torch.log_softmax(_ref / 2, dim=-1).max().item()
            log_prob_list.append(cur_log_prob)
            if not next_tokens[0].item() in value_indices:
                if len(sub_value_scores) > 0:
                    sub_value_scores = torch.stack(sub_value_scores, dim=1)
                    value_scores.append(sub_value_scores)
                sub_value_scores = []
            else:
                sub_value_scores.append(_ref[:, value_indices].detach().clone())
            cur_topk_score, cur_topk_index = torch.topk(_ref, topk, dim=-1)
            topk_scores.append(cur_topk_score)
            topk_indices.append(cur_topk_index)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        del outputs
    
    if return_scores:
        topk_scores = torch.cat(topk_scores, dim=0).tolist()
        topk_indices = torch.cat(topk_indices, dim=0).tolist()
        return dict(
            output_ids=input_ids,
            topk_scores=topk_scores,
            topk_indices=topk_indices,
        )
    else:
        return dict(
            output_ids=input_ids,
        )