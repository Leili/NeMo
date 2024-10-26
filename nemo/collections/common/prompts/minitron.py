from collections import defaultdict

from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.prompts import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import BOS_SLOT, EOS_SLOT, Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.utils import logging


class MinitronPromptFormatter(PromptFormatter):
    """
    Prompt formatter similar to llama3 example
    """
    NAME = "minitron"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "system": {
            "template": f"<extra_id_0>System\nYou are an AI assistant who can understand and generate multimodal content, including text, speech and audio. Please recognize the input audio and give appropriate reply in text.\n|message|\n\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"<extra_id_1>User\n|message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"<extra_id_1>Assistant\n|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

@registered_prompt_format_fn
def minitron(cuts: CutSet, tokenizer: TokenizerSpec):
    logging.info(f"LEILI DEBUG")
    logging.info(f"HELLO FROM MINITRON")
    prompt = MinitronPromptFormatter(tokenizer)
    ans = defaultdict(list)
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut.first_non_padding_cut
        if cut.has_custom("context"):
            context = cut.context
        else:
            context = cut.default_context


        audio_locator_tag = "<extra_id_10><extra_id_12><extra_id_11>"

        turns = []
        turns.append({"role": "system", "slots": {"message": context}})
        turns.append({"role": "user", "slots": {"message": audio_locator_tag}})

        if (text := cut.supervisions[0].text) is not None:
            turns.append({"role": "assistant", "slots": {"message": text}})

        for k, v in prompt.encode_dialog(turns).items():
            ans[k].append(v)

    return ans