from dataclasses import dataclass, fields, asdict, replace

from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig, NemotronHForCausalLM

if __name__ == "__main__":
    # Get base nemotronH config from HF
    config = AutoConfig.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)

    # Load custom nemotronH yaml
    cfg = OmegaConf.load('/home/jasoli/gitrepos/NeMo/examples/tts/conf/magpietts/easy_magpietts_lhotse.yaml')
    cfg = cfg.model
    # Build config from YAML parameters
    nemotron_h_config_dict = dict(cfg.get('nemotron_h_config', {}))
    # Ensure hidden_size matches embedding_dim for compatibility
    if 'hidden_size' not in nemotron_h_config_dict:
        nemotron_h_config_dict['hidden_size'] = cfg.embedding_dim

    # Merge configs
    custom_config = replace(config, **nemotron_h_config_dict)
    print("########"*10)
    print(custom_config)
    print("########"*10)

    # Create model
    nemotronh = NeMoAutoModelForCausalLM.from_config(custom_config)
    import ipdb; ipdb.set_trace()
    print(nemotronh)

    # nemotron_config = asdict(NemotronHConfig(**nemotron_h_config_dict))
    # print(nemotron_config)
    # config = AutoConfig.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)
    # print("########"*10)
    # print(config)
    # nemotron_config["architectures"] = config.architectures
    # # nemotron_config.auto_map = config.auto_map
    # print("########"*10)
    # # print(set([f.name for f in fields(config)])-set(([f.name for f in fields(nemotron_config)])))
    # print(set([f.name for f in fields(config)])-set(nemotron_config.keys()))
    # print("########"*10)
    # import ipdb; ipdb.set_trace()
    # print(nemotron_config)
    # nemotronh = NeMoAutoModelForCausalLM.from_config(nemotron_config)
    # # nemotronh = NeMoAutoModelForCausalLM.from_config(config)
    # print(nemotronh)
    # print("########"*10)
    # # print()