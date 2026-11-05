// bubble_prepass for vsTerrainMeshDepthPre, whose output signature is
// SV_Position + TEXCOORD15 (uint, register 1). Mirrors it so the pipeline links.
#define BUBBLE_USE_SCENE_CONSTANTS
#include "./bubble.hlsli"

void main(float4 sv_position : SV_Position, nointerpolation uint unused : TEXCOORD15) {
  if (BubbleShouldCut(sv_position)) discard;
}
