#ifndef SRC_SEETHROUGHBUBBLE_BUBBLE_HLSLI_
#define SRC_SEETHROUGHBUBBLE_BUBBLE_HLSLI_

#include "./shared.h"

// Stable per-pixel noise (Jimenez 2014). It doesn't change over time, so
// temporal upscalers resolve the dither into a soft edge instead of shimmer.
float BubbleInterleavedGradientNoise(float2 pixel) {
  return frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f))));
}

// Core test. uv: 0..1 screen position; aspect: width / height;
// view_distance: distance from the camera in metres; cut_distance: cut what is closer.
bool BubbleShouldCutAt(float2 pixel, float2 uv, float2 center, float aspect, float view_distance, float cut_distance,
                       float radius) {
  if (shader_injection.bubble_enabled == 0.f) return false;

  float2 offset_from_center = uv - center;
  float width_scale = max(shader_injection.bubble_width_scale, 0.05f);  // oval: width / height
  float distance_in_radii = length(offset_from_center * float2(aspect / width_scale, 1.f)) / radius;
  if (!(distance_in_radii < 1.f)) return false;

  if (shader_injection.debug_ignore_depth == 0.f && !(view_distance < cut_distance)) return false;

  float cut_strength = saturate((1.f - distance_in_radii) / max(shader_injection.bubble_softness, 1e-3f));
  return BubbleInterleavedGradientNoise(pixel) < cut_strength;
}

#ifdef BUBBLE_USE_SCENE_CONSTANTS
// The game's own SceneConstantBuffer, as bound for the depth-prepass vertex
// shaders (b19, space35). Only the fields we read are declared, at their offsets.
cbuffer BubbleSceneConstants : register(b19, space35) {
  float4 bubble_scene_buffer_size_and_inv_size : packoffset(c4);                // _bufferSizeAndInvSize (offset 64)
  float4 bubble_scene_view_pos : packoffset(c8);                                // _viewPos (128), xyz
  column_major float4x4 bubble_scene_view_proj_relative : packoffset(c18);      // _viewProjRelative (288)
  column_major float4x4 bubble_scene_inv_view_proj_relative : packoffset(c30);  // _invViewProjRelative (480)
}

// Bubble center (uv) and the character's raw depth. With the game's player position
// (player_world_valid) they are projected here with this frame's camera; otherwise
// they are the add-on's screen values.
void BubblePlayerScreen(out float2 center, out float depth) {
  float3 player_world = float3(shader_injection.player_world_x, shader_injection.player_world_y,
                               shader_injection.player_world_z);
  float4 clip = mul(bubble_scene_view_proj_relative, float4(player_world - bubble_scene_view_pos.xyz, 1.f));
  float inv_w = 1.f / max(clip.w, 1e-6f);
  bool use_world = and(shader_injection.player_world_valid != 0.f, clip.w > 1e-6f);
  center.x = select(use_world, clip.x * inv_w * 0.5f + 0.5f, shader_injection.bubble_center_u);
  center.y = select(use_world, 0.5f - clip.y * inv_w * 0.5f, shader_injection.bubble_center_v);
  depth = select(use_world, clip.z * inv_w, shader_injection.player_depth);
}

// Camera-relative position of the point at screen uv with raw depth z,
// reconstructed like the game's tree shaders do.
float3 BubbleRelativePosition(float2 uv, float z) {
  float3 ndc = float3(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, z);
  float4 relative = mul(bubble_scene_inv_view_proj_relative, float4(ndc, 1.f));
  return relative.xyz / relative.w;
}

float BubbleViewDistance(float2 uv, float z) {
  return length(BubbleRelativePosition(uv, z));
}

// Exact: per-pixel distance and, when tracking, the character's distance.
bool BubbleShouldCut(float4 sv_position) {
  float2 inv_size = bubble_scene_buffer_size_and_inv_size.zw;
  float2 uv = sv_position.xy * inv_size;
  float view_distance = BubbleViewDistance(uv, sv_position.z);
  float2 center;
  float projected_depth;
  BubblePlayerScreen(center, projected_depth);
  float player_depth = max(projected_depth, 1e-7f);
  float3 player_position = BubbleRelativePosition(center, player_depth);
  float player_distance = length(player_position);
  bool tracking = shader_injection.player_depth > 0.f;
  float cut_distance = tracking ? player_distance - shader_injection.cut_margin : shader_injection.bubble_cut_distance;
  // Metres per unit of screen height at the character's depth, so the bubble
  // keeps its size around the character when the camera zooms (distance or FOV).
  float metres_per_height = length(BubbleRelativePosition(center + float2(0.f, 0.01f), player_depth) - player_position) / 0.01f;
  float radius = (tracking && shader_injection.bubble_world_radius > 0.f)
                     ? shader_injection.bubble_world_radius / max(metres_per_height, 1e-4f)
                     : shader_injection.bubble_radius;
  float aspect = bubble_scene_buffer_size_and_inv_size.x * inv_size.y;
  return BubbleShouldCutAt(sv_position.xy, uv, center, aspect, view_distance, cut_distance, radius);
}
#else
// Fallback: render size from the add-on (bound viewport), distance from the
// near plane setting, assuming reversed-Z with an infinite far plane.
bool BubbleShouldCut(float4 sv_position) {
  float2 render_size = float2(shader_injection.render_width, shader_injection.render_height);
  float2 uv = sv_position.xy / render_size;
  float view_distance = shader_injection.camera_near / max(sv_position.z, 1e-7f);
  float cut_distance = shader_injection.player_depth > 0.f
                           ? shader_injection.camera_near / shader_injection.player_depth - shader_injection.cut_margin
                           : shader_injection.bubble_cut_distance;
  float2 center = float2(shader_injection.bubble_center_u, shader_injection.bubble_center_v);
  return BubbleShouldCutAt(sv_position.xy, uv, center, render_size.x / render_size.y, view_distance, cut_distance,
                           shader_injection.bubble_radius);
}
#endif

#endif  // SRC_SEETHROUGHBUBBLE_BUBBLE_HLSLI_
