import torch
from PIL import Image
import numpy as np
from config import Config
from diffusers import StableDiffusionPipeline
from utils import prepare_bboxes, compute_cross_loss, compute_self_loss, AttentionStore,  hook_attention_processors

torch.backends.cuda.matmul.allow_tf32 = True


def update_latent(latents, loss, step_size):
  grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
  latents = latents - step_size * grad_cond

  return latents

def decode_latents(latents, decoder):
  latents = 1 / 0.18215 * latents
  with torch.no_grad():
    image = decoder.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype('uint8')
  pil_images = [Image.fromarray(image) for image in images]

  return pil_images

def generate(pipeline, attention_store, prompt, indices, fg_bounding_boxes, bg_bounding_boxes, config, seed, device):

  generator = torch.manual_seed(seed)

  height = config.height
  width = config.width
  num_inference_steps = config.num_inference_steps
  guidance_scale = config.guidance_scale
  max_iter_to_alter = config.max_iter_to_alter
  topk_coef = config.topk_coef
  dropout = config.dropout
  margin = config.margin

  text_input = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")

  with torch.no_grad():
    cond_text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]

  max_length = text_input.input_ids.shape[-1]
  uncond_input = pipeline.tokenizer(
      [''] * config.batch_size, padding='max_length', max_length=max_length, return_tensors='pt'
  )
  with torch.no_grad():
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(device))[0]

  text_embeddings = torch.cat([uncond_embeddings, cond_text_embeddings])

  latents = torch.randn(
    (config.batch_size, pipeline.unet.in_channels, height // 8, width // 8),
    generator=generator,
  )
  latents = latents.to(device)
  pipeline.scheduler.set_timesteps(num_inference_steps)

  scale_range = np.linspace(1.0, 0.5, len(pipeline.scheduler.timesteps))
  step_size = config.scale_factor * np.sqrt(scale_range)

  bbox_masks = prepare_bboxes(fg_bounding_boxes, bg_bounding_boxes, config.target_res, device)

  for i, t in enumerate(pipeline.scheduler.timesteps):
    if i < max_iter_to_alter:
      for j in range(config.refinement_steps):
        with torch.enable_grad():
          latents = latents.clone().detach().requires_grad_(True)
          pipeline.unet(latents,t,encoder_hidden_states=cond_text_embeddings).sample
          pipeline.unet.zero_grad()

          cross_attentions, self_attentions = attention_store.aggregate_attention(from_where=(['up']),)

          loss = compute_cross_loss(cross_attentions, indices, bbox_masks, topk_coef, dropout, margin) + compute_self_loss(self_attentions, indices, bbox_masks, topk_coef, dropout)

          latents = update_latent(latents, loss, step_size[i])
          torch.cuda.empty_cache()

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    cross_attention_kwargs = {'timestep': i, 'max_iter_to_alter': max_iter_to_alter, 'bbox_masks': bbox_masks, 'indices': indices}

    with torch.no_grad():
      noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
    torch.cuda.empty_cache()

  img = decode_latents(latents, pipeline.vae)[0]

  return img

def main():
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
    attention_store = AttentionStore(config.target_res)
    hook_attention_processors(pipeline, attention_store)

    prompt = ''
    # Attending indices for foreground objects
    fg_indices = []
    # Attending indices for backgrounds
    bg_indices = []
    # Bounding box coordinates for foreground objects
    fg_bounding_boxes = []
    # Bounding box coordinates for backgrounds, leave empty if no background is specified
    bg_bounding_boxes = []
    image = generate(pipeline, attention_store, prompt, fg_indices + bg_indices, fg_bounding_boxes, bg_bounding_boxes, config, 8888, device)
    image.save(f'/images/1.jpg')

if __name__ == "__main__":
    main()