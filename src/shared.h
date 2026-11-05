#ifndef SRC_SEETHROUGHBUBBLE_SHARED_H_
#define SRC_SEETHROUGHBUBBLE_SHARED_H_

// Values pushed by the add-on into every patched/injected shader.
// Must be 32-bit aligned; keep the size a multiple of 16 bytes.
struct ShaderInjectData {
  float bubble_enabled;
  float bubble_center_u;     // player's screen position, 0..1
  float bubble_center_v;
  float bubble_radius;       // fraction of screen height
  float bubble_softness;     // 0..1, width of the dithered edge relative to the radius
  float bubble_cut_distance; // metres from camera; only fragments closer than this are cut
  float camera_near;         // near plane, for linearizing reversed-Z depth
  float render_width;        // render (pre-upscale) resolution in pixels
  float render_height;
  float player_depth;        // tracked character depth (raw reversed-Z) at the bubble center; 0 = use cut distance
  float debug_ignore_depth;  // cut everything inside the circle, ignoring distance
  float cut_margin;          // metres kept in front of the tracked character
  float bubble_world_radius; // auto mode: bubble radius in metres around the character (0 = use bubble_radius)
  float bubble_width_scale;  // bubble width / height (1 = circle)
  // Auto mode with the game's player position: the point to follow in absolute world
  // coordinates (metres, Y up), projected in the shader with the frame's own camera
  // (no readback lag). player_world_valid = 0: use bubble_center_u/v and player_depth.
  float player_world_valid;
  float player_world_x;
  float player_world_y;
  float player_world_z;
  float padding1;
  float padding2;
};

#ifndef __cplusplus
// b0, space73: unique to this add-on (RenoDX's Crimson Desert HDR mod uses b13, space50).
cbuffer shader_injection : register(b0, space73) {
  ShaderInjectData shader_injection : packoffset(c0);
}
#endif

#endif  // SRC_SEETHROUGHBUBBLE_SHARED_H_
