# Per-layer tables for the mentor

QuIP# sigma_reg actually used per layer (default 0.01); layers that needed a retry (10x = 0.1, 100x = 1.0):
{
 "quip_w2": {
  "backbone": {
   "backbone.model.model.language_model.layers.2.mlp.down_proj": 0.1
  }
 },
 "quip10_w2": {
  "backbone": {
   "backbone.model.model.language_model.layers.2.mlp.down_proj": 0.1
  }
 }
}

qtip_w2 / backbone: mean output error 0.0046; worst 5: visual.blocks.11.mlp.linear_fc1 0.014; visual.blocks.17.attn.qkv 0.014; visual.blocks.20.attn.qkv 0.014; visual.blocks.19.attn.qkv 0.014; visual.blocks.12.mlp.linear_fc1 0.013

qtip_w2 / head_libero_spatial: mean output error 0.0038; worst 5: transformer_blocks.8.attn1.to_v 0.019; transformer_blocks.4.attn1.to_v 0.018; transformer_blocks.12.attn1.to_v 0.018; transformer_blocks.16.attn1.to_v 0.018; transformer_blocks.20.attn1.to_v 0.018

qtip_w2 / head_libero_object: mean output error 0.0035; worst 5: transformer_blocks.8.attn1.to_v 0.018; transformer_blocks.4.attn1.to_v 0.017; transformer_blocks.12.attn1.to_v 0.017; transformer_blocks.16.attn1.to_v 0.017; transformer_blocks.20.attn1.to_v 0.017

qtip_w2 / head_libero_goal: mean output error 0.0039; worst 5: transformer_blocks.8.attn1.to_v 0.020; transformer_blocks.0.attn1.to_v 0.019; transformer_blocks.4.attn1.to_v 0.019; transformer_blocks.12.attn1.to_v 0.019; transformer_blocks.16.attn1.to_v 0.019

qtip_w2 / head_libero_10: mean output error 0.0039; worst 5: transformer_blocks.8.attn1.to_v 0.020; transformer_blocks.0.attn1.to_v 0.019; transformer_blocks.4.attn1.to_v 0.019; transformer_blocks.12.attn1.to_v 0.019; transformer_blocks.16.attn1.to_v 0.019

quip_w2 / backbone: mean output error 0.0062; worst 5: visual.blocks.11.mlp.linear_fc1 0.020; visual.blocks.17.attn.qkv 0.019; visual.blocks.19.attn.qkv 0.018; visual.blocks.20.attn.qkv 0.018; visual.blocks.12.mlp.linear_fc1 0.018

quip_w2 / head_libero_spatial: mean output error 0.0051; worst 5: transformer_blocks.8.attn1.to_v 0.026; transformer_blocks.4.attn1.to_v 0.025; transformer_blocks.12.attn1.to_v 0.025; transformer_blocks.16.attn1.to_v 0.024; transformer_blocks.0.attn1.to_v 0.024

quip_w2 / head_libero_object: mean output error 0.0047; worst 5: transformer_blocks.8.attn1.to_v 0.024; transformer_blocks.4.attn1.to_v 0.023; transformer_blocks.12.attn1.to_v 0.023; transformer_blocks.16.attn1.to_v 0.023; transformer_blocks.20.attn1.to_v 0.023

quip_w2 / head_libero_goal: mean output error 0.0052; worst 5: transformer_blocks.8.attn1.to_v 0.027; transformer_blocks.0.attn1.to_v 0.026; transformer_blocks.4.attn1.to_v 0.026; transformer_blocks.12.attn1.to_v 0.026; transformer_blocks.16.attn1.to_v 0.025

quip_w2 / head_libero_10: mean output error 0.0053; worst 5: transformer_blocks.8.attn1.to_v 0.027; transformer_blocks.0.attn1.to_v 0.026; transformer_blocks.4.attn1.to_v 0.026; transformer_blocks.12.attn1.to_v 0.026; transformer_blocks.16.attn1.to_v 0.025

quip10_w2 / backbone: mean output error 0.0074; worst 5: layers.13.self_attn.o_proj 0.026; layers.15.self_attn.v_proj 0.023; visual.blocks.11.mlp.linear_fc1 0.019; visual.blocks.17.attn.qkv 0.019; visual.blocks.19.attn.qkv 0.018

quip10_w2 / head_libero_spatial: mean output error 0.0058; worst 5: transformer_blocks.8.attn1.to_v 0.025; transformer_blocks.4.attn1.to_v 0.024; transformer_blocks.12.attn1.to_v 0.024; transformer_blocks.16.attn1.to_v 0.024; transformer_blocks.0.attn1.to_v 0.023

quip10_w2 / head_libero_object: mean output error 0.0054; worst 5: transformer_blocks.8.attn1.to_v 0.023; transformer_blocks.4.attn1.to_v 0.023; transformer_blocks.12.attn1.to_v 0.023; transformer_blocks.16.attn1.to_v 0.022; transformer_blocks.20.attn1.to_v 0.022

quip10_w2 / head_libero_goal: mean output error 0.0059; worst 5: transformer_blocks.8.attn1.to_v 0.026; transformer_blocks.0.attn1.to_v 0.025; transformer_blocks.4.attn1.to_v 0.025; transformer_blocks.12.attn1.to_v 0.025; transformer_blocks.16.attn1.to_v 0.025

quip10_w2 / head_libero_10: mean output error 0.0059; worst 5: transformer_blocks.8.attn1.to_v 0.026; transformer_blocks.0.attn1.to_v 0.026; transformer_blocks.4.attn1.to_v 0.025; transformer_blocks.12.attn1.to_v 0.025; transformer_blocks.16.attn1.to_v 0.024
