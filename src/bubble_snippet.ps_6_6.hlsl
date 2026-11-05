// Source of the IR that the add-on splices into game pixel shaders (trees).
// Embedded as DXIL by the build; at runtime the add-on disassembles it, renames its values and
// remaps the SceneConstantBuffer binding and SV_Position input id.
// Must stay branch-free (straight-line) so it can be inserted at the entry.
#include "./shared.h"

cbuffer BubbleSceneConstants : register(b19, space35) {
  float4 bubble_scene_buffer_size_and_inv_size : packoffset(c4);                // _bufferSizeAndInvSize
  float4 bubble_scene_view_pos : packoffset(c8);                                // _viewPos (xyz)
  column_major float4x4 bubble_scene_view_proj_relative : packoffset(c18);      // _viewProjRelative
  column_major float4x4 bubble_scene_inv_view_proj_relative : packoffset(c30);  // _invViewProjRelative
}

void main(float4 sv_position : SV_Position) {
  float2 inv_size = bubble_scene_buffer_size_and_inv_size.zw;
  float2 uv = sv_position.xy * inv_size;
  float3 ndc = float3(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, sv_position.z);
  float4 relative = mul(bubble_scene_inv_view_proj_relative, float4(ndc, 1.f));
  float view_distance = length(relative.xyz / relative.w);
  float aspect = bubble_scene_buffer_size_and_inv_size.x * inv_size.y;

  // Tracked character distance (same reconstruction at the bubble center).
  // With the game's player position: project it with this frame's camera (no readback lag).
  float3 player_world = float3(shader_injection.player_world_x, shader_injection.player_world_y,
                               shader_injection.player_world_z);
  float4 player_clip = mul(bubble_scene_view_proj_relative, float4(player_world - bubble_scene_view_pos.xyz, 1.f));
  float inv_w = 1.f / max(player_clip.w, 1e-6f);
  bool use_world = and(shader_injection.player_world_valid != 0.f, player_clip.w > 1e-6f);
  float2 center = float2(select(use_world, player_clip.x * inv_w * 0.5f + 0.5f, shader_injection.bubble_center_u),
                         select(use_world, 0.5f - player_clip.y * inv_w * 0.5f, shader_injection.bubble_center_v));
  float player_depth = max(select(use_world, player_clip.z * inv_w, shader_injection.player_depth), 1e-7f);
  float3 player_ndc = float3(center.x * 2.f - 1.f, 1.f - center.y * 2.f, player_depth);
  float4 player_relative = mul(bubble_scene_inv_view_proj_relative, float4(player_ndc, 1.f));
  float3 player_position = player_relative.xyz / player_relative.w;
  float player_distance = length(player_position);
  // Metres per unit of screen height at the character's depth (bubble keeps its world size when zooming).
  float3 above_ndc = float3(player_ndc.x, player_ndc.y - 0.02f, player_depth);
  float4 above_relative = mul(bubble_scene_inv_view_proj_relative, float4(above_ndc, 1.f));
  float metres_per_height = length(above_relative.xyz / above_relative.w - player_position) / 0.01f;
  bool tracking = shader_injection.player_depth > 0.f;
  float radius = select(and(tracking, shader_injection.bubble_world_radius > 0.f),
                        shader_injection.bubble_world_radius / max(metres_per_height, 1e-4f), shader_injection.bubble_radius);
  // select(), not ?:, which short-circuits in HLSL 2021 and would add a branch.
  float cut_distance = select(shader_injection.player_depth > 0.f, player_distance - shader_injection.cut_margin,
                              shader_injection.bubble_cut_distance);

  float2 offset = uv - center;
  float width_scale = max(shader_injection.bubble_width_scale, 0.05f);  // oval: width / height
  float distance_in_radii = length(offset * float2(aspect / width_scale, 1.f)) / radius;
  float cut_strength = saturate((1.f - distance_in_radii) / max(shader_injection.bubble_softness, 1e-3f));
  float noise = frac(52.9829189f * frac(dot(sv_position.xy, float2(0.06711056f, 0.00583715f))));

  bool enabled = shader_injection.bubble_enabled != 0.f;
  bool near_enough = or(shader_injection.debug_ignore_depth != 0.f, view_distance < cut_distance);
  bool cut = and(and(enabled, distance_in_radii < 1.f), and(near_enough, noise < cut_strength));
  if (cut) discard;
}
