// Pixel shader we ADD to clones of the engine's depth-only prepass pipelines
// (vsStandardDepthPre), which have no pixel shader. It writes no color; it only
// discards so the occluder leaves no depth inside the bubble.
// Reads the game's SceneConstantBuffer for exact camera distances.
#define BUBBLE_USE_SCENE_CONSTANTS
#include "./bubble.hlsli"

void main(float4 sv_position : SV_Position) {
  if (BubbleShouldCut(sv_position)) discard;
}
