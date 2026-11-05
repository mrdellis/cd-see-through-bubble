// Fallback bubble_prepass_terrain without the game's SceneConstantBuffer.
#include "./bubble.hlsli"

void main(float4 sv_position : SV_Position, nointerpolation uint unused : TEXCOORD15) {
  if (BubbleShouldCut(sv_position)) discard;
}
