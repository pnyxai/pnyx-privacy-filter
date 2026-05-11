import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        max_seq_str = params.get("MAX_SEQUENCE_LENGTH", {}).get("string_value", "")
        if not max_seq_str:
            raise ValueError("MAX_SEQUENCE_LENGTH is not defined")
        try:
            self.max_length = int(max_seq_str)
        except (ValueError, TypeError):
            raise ValueError("Invalid value for MAX_SEQUENCE_LENGTH")

        self.tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                prompts = [
                    x.decode("utf-8") for x in in_tensor.as_numpy().flatten()
                ]

                encoded = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np"
                )
                input_ids_np = encoded["input_ids"].astype(np.int64)
                attn_mask_np = encoded["attention_mask"].astype(np.int64)

                out_0 = pb_utils.Tensor("INPUT_IDS", input_ids_np)
                out_1 = pb_utils.Tensor("ATTN_MASK", attn_mask_np)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_0, out_1])
                )
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def finalize(self):
        pass
