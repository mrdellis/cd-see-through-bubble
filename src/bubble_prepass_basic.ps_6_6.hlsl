// Fallback bubble_prepass without the game's SceneConstantBuffer (used if the
// prepass root signature doesn't expose it to pixel shaders).
#include "./bubble.hlsli"

void main(float4 sv_position : SV_Position) {
  if (BubbleShouldCut(sv_position)) discard;
}
