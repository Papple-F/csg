import torch
import math
from diffusers.models.attention_processor import Attention



class AttentionStore:

  def __init__(self, target_res):
    """
    Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
    process
    """
    self.num_att_layers = -1
    self.cur_att_layer = 0
    self.target_res = target_res
    self.cross_step_store = self.get_empty_store()
    self.self_step_store = self.get_empty_store()
    self.cross_attention_store = {}
    self.self_attention_store = {}

  def get_empty_store(self):
    empty_store = {"down": [], "mid": [], "up": []}

    return empty_store

  def store_attention_probs(self, attn, is_cross: bool, place_in_unet: str):
    if self.cur_att_layer >= 0 and self.target_res == math.sqrt(attn.shape[1]):
      if is_cross:
        self.cross_step_store[place_in_unet].append(attn)
      else:
        self.self_step_store[place_in_unet].append(attn)

    self.cur_att_layer += 1
    if self.cur_att_layer == self.num_att_layers:
      self.cur_att_layer = 0
      self.between_steps_probs()

  def between_steps_probs(self):
    self.cross_attention_store = self.cross_step_store
    self.self_attention_store = self.self_step_store
    self.cross_step_store = self.get_empty_store()
    self.self_step_store = self.get_empty_store()

  def aggregate_attention(self, from_where):
    cross_attention_maps, self_attention_maps = self.cross_attention_store, self.self_attention_store
    cross_attns = []
    for location in from_where:
      for cross_attn in cross_attention_maps[location]:
        cross_attns.append(cross_attn)
    cross_attns = torch.cat(cross_attns, dim=0)
    cross_attns = cross_attns.sum(0) / cross_attns.shape[0]

    self_attns = []
    for location in from_where:
      for self_attn in self_attention_maps[location]:
        self_attns.append(self_attn)
    self_attns = torch.cat(self_attns, dim=0)
    self_attns = self_attns.sum(0) / self_attns.shape[0]

    return cross_attns, self_attns

class ARAttnProcessor:
  def __init__(self, attnstore, place_in_unet):
    super().__init__()
    self.attnstore = attnstore
    self.place_in_unet = place_in_unet

  def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, timestep=0, max_iter_to_alter=0, target_res=16,
               bbox_masks=None, indices=None):
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    query = attn.to_q(hidden_states)

    is_cross = encoder_hidden_states is not None
    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    # Attention redistribution
    if not attention_probs.requires_grad and is_cross and timestep <= max_iter_to_alter:
      if math.sqrt(attention_probs.shape[1]) == target_res:
        accumulated_attentions = {}
        for index in indices:
          in_box_mask = bbox_masks[indices.index(index)]
          accumulated_attention_per_token = []
          for token_index in indices:
            accumulated_attention_per_token.append(attention_probs[:,:,token_index] * in_box_mask)
          accumulated_attention_per_token = torch.stack(accumulated_attention_per_token, dim=-1).sum(-1)
          accumulated_attentions[index] = accumulated_attention_per_token
        for index in indices:
          attention_probs[:,:,index] = accumulated_attentions[index]
        attention_probs[:,:,:] /= attention_probs[:,:,:].sum(-1, keepdim=True)

    self.attnstore.store_attention_probs(attention_probs, is_cross, self.place_in_unet)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states

def fit_bboxes(bboxes_coords, target_res) :
  # Adapt bounding box coordinates to match a target attention dimension
  fitted_bboxes = []
  for i in range(len(bboxes_coords)):
    fitted_bboxes += [[max(round(bbox / (512 / target_res)), 0) for bbox in bboxes_coords[i]]]

  return fitted_bboxes

def create_bbox_masks(bboxes, target_res, device, front_layer = None):
  # Create bounding box masks that do not overlap with upper-level masks
  fitted_bbox_masks = []

  for bbox in bboxes:
    mask = torch.zeros((target_res, target_res))
    x1, y1, x2, y2 = bbox
    ones_mask = torch.ones([y2 - y1, x2 - x1])
    mask[y1:y2, x1:x2] = ones_mask
    if front_layer is not None:
      mask -= front_layer
      mask = torch.where(mask == 1, 1, 0)
    fitted_bbox_masks.append(mask.view(-1).to(device))

  return fitted_bbox_masks

def prepare_bboxes(fg_bboxes, bg_bboxes, target_res, device):
  # Create bounding box masks for target resolution
  fg_bboxes = fit_bboxes(fg_bboxes, target_res)
  fg_bbox_masks = create_bbox_masks(fg_bboxes, target_res, device)

  if len(bg_bboxes) != 0:
    bg_bboxes = fit_bboxes(bg_bboxes, target_res)
    front_layers = []
    masks = fg_bbox_masks
    layered_mask = torch.stack(masks, dim=-1).sum(-1)
    layered_mask = torch.where(layered_mask >= 1, 1, 0)
    front_layers = layered_mask.reshape(target_res, target_res).detach().cpu()

    bg_bbox_masks = create_bbox_masks(bg_bboxes, target_res, device, front_layers)

    bbox_masks = fg_bbox_masks + bg_bbox_masks
  else:
    bbox_masks = fg_bbox_masks

  return bbox_masks

def compute_self_loss(self_attentions, indices, bbox_masks, topk_coef, dropout):

  within_token_losses = []

  for index in indices:
    in_box_mask = bbox_masks[indices.index(index)]
    out_box_mask = (1 - in_box_mask)
    pixel_indices = torch.nonzero(in_box_mask).flatten().tolist()

    attention_per_token = []
    for pixel_index in pixel_indices:
      attention_per_token.append(self_attentions[:,pixel_index])
    attention_per_token = torch.stack(attention_per_token, dim=-1).sum(-1)

    in_box_attention = in_box_mask * attention_per_token
    out_box_attention = out_box_mask * attention_per_token

    k = max(torch.tensor(1), (in_box_mask.sum() * topk_coef).long())
    topk_values, topk_indices = (in_box_attention).topk(k)
    r = max(torch.tensor(1), int(torch.round(k*dropout).item()))
    in_box_topk_indices = torch.randperm(topk_indices.shape[0])[:r]
    rand_topk_values = topk_values[in_box_topk_indices]
    rand_topk_indces = topk_indices[in_box_topk_indices]
    in_box_rand_attention = rand_topk_values.sum()

    topk_values, neg_topk_indices = (out_box_attention).topk(r)
    out_box_rand_attention_sum = topk_values.sum()

    within_token_loss = (1 - (in_box_rand_attention / (in_box_rand_attention + out_box_rand_attention_sum))) ** 2
    within_token_losses.append(within_token_loss)

  latent_loss = 5 * sum(within_token_losses)

  return latent_loss

def compute_cross_loss(cross_attentions, indices, bbox_masks, topk_coef, dropout, margin):

  within_token_losses = []
  across_token_losses = []

  for index in indices:
    attention_per_token = cross_attentions[:,index]
    in_box_mask = bbox_masks[indices.index(index)]
    out_box_mask = (1 - in_box_mask)
    in_box_attention = in_box_mask * attention_per_token
    out_box_attention = out_box_mask * attention_per_token

    # Selective sampling
    k = max(torch.tensor(1), (in_box_mask.sum() * topk_coef).long())
    topk_values, topk_indices = (in_box_attention).topk(k)
    r = max(torch.tensor(1), int(torch.round(k*dropout).item()))
    in_box_topk_indices = torch.randperm(topk_indices.shape[0])[:r]
    rand_topk_values = topk_values[in_box_topk_indices]
    rand_topk_indces = topk_indices[in_box_topk_indices]
    in_box_rand_attention = rand_topk_values.sum()
    in_box_rand_attention_mean = rand_topk_values.mean()

    topk_values, neg_topk_indices = (out_box_attention).topk(r)
    out_box_rand_attention_sum = topk_values.sum()

    within_token_loss = (1 - (in_box_rand_attention / (in_box_rand_attention + out_box_rand_attention_sum))) ** 2
    within_token_losses.append(within_token_loss)

    if len(indices) > 1:
      in_box_attention_across_tokens = []
      for other_index in indices:
        if index != other_index:
          in_box_rand_attention_token = (in_box_mask * cross_attentions[:,other_index])[rand_topk_indces].mean()
          in_box_attention_across_tokens.append(in_box_rand_attention_token)
      max_in_box_attention = max(in_box_attention_across_tokens)

      across_token_loss = (min(in_box_rand_attention_mean - margin, max_in_box_attention) - max_in_box_attention) ** 2
      across_token_losses.append(across_token_loss)

  latent_loss = 5 * (sum(within_token_losses) + 50 * sum(across_token_losses))

  return latent_loss

def hook_attention_processors(pipeline, attention_store):

  attn_procs = {}
  cross_att_count = 0
  for name in pipeline.unet.attn_processors.keys():
    if name.startswith("mid_block"):
      place_in_unet = "mid"
    elif name.startswith("up_blocks"):
      place_in_unet = "up"
    elif name.startswith("down_blocks"):
      place_in_unet = "down"
    else:
      continue
    cross_att_count += 1
    attn_procs[name] = ARAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)

  pipeline.unet.set_attn_processor(attn_procs)
  attention_store.num_att_layers = cross_att_count