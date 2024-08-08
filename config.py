from dataclasses import dataclass

@dataclass
class Config:
  height: int = 512
  width: int = 512
  num_inference_steps: int = 50
  batch_size: int = 1
  guidance_scale: float = 7.5
  topk_coef: float = 0.8
  margin: float = 0.1
  dropout:float = 0.5
  max_iter_to_alter: int = 25
  refinement_steps: int = 5
  scale_factor: float = 0.5
  target_res: int = 16