import torch
import hydra

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_rl_model import LoRAMergeConfig

cs = ConfigStore.instance()
cs.store(name="config/lora_merge", node=LoRAMergeConfig)

@hydra.main(config_path="config/lora_merge", version_base=None)
def main(cfg: LoRAMergeConfig):
    default_config = OmegaConf.structured(LoRAMergeConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    base_model_path = cfg.base_model
    adapter_path = cfg.adapter_path
    output_path = cfg.output_path

    device_arg = {"device_map": "auto"}
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )
    print(f"Loading and merging PEFT from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__=="__main__":
    main()