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

    INSERT_BOS = True
    INSERT_EOS = True

    TEMPLATE = {
        "system": {
            "template": f"System\nYou are an AI assistant who can understand and generate multimodal content, including text, speech and audio. Please recognize the input audio and give appropriate reply in text.\n|message|</s>\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"<s>User\n|message|</s>\n<s>Assistant\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|</s>\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

#NOTE: THIS FUNCTION IS NOT USED IN TEMPLATE CONSTRUCTION
@registered_prompt_format_fn
def minitron(cuts: CutSet, tokenizer: TokenizerSpec):
    prompt = MinitronPromptFormatter(tokenizer)
    ans = defaultdict(list)
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut.first_non_padding_cut
        if cut.has_custom("context"):
            context = cut.context
        else:
            context = cut.default_context


        audio_locator_tag = "<SPECIAL_14><SPECIAL_16><SPECIAL_15>"

        turns = []
        turns.append({"role": "system", "slots": {"message": context}})
        turns.append({"role": "user", "slots": {"message": audio_locator_tag}})

        if (text := cut.supervisions[0].text) is not None:
            turns.append({"role": "assistant", "slots": {"message": text}})

        for k, v in prompt.encode_dialog(turns).items():
            ans[k].append(v)

    return ans