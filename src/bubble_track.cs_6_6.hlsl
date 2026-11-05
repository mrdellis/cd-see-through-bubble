// Character tracking (Phase 4). Runs once per frame on ReShade's command list
// over our copy of the game's scene depth-stencil (taken at the next depth
// clear, so it holds the previous frame).
//
// mode 0: clear the output. mode 1: scan a circular window around the hint
// position. Character pixels (stencil & write_mask is in the learned set of
// character material IDs) are binned by stencil value (slot = rank of the value
// in the learned set) and by depth. The CPU then picks the group nearest to the
// home point, merges nearby groups (one character is drawn with several stencil
// values) and takes their centroid and front-most depth. Binning by value keeps
// props that the game draws with character pipelines (e.g. clay jars) apart from
// the character even when they stand at the same depth.

struct TrackConstants {
  uint2 size;          // depth texture size in pixels
  int2 box_min;        // top-left of the scanned box
  float2 hint_px;      // search center in pixels
  float radius_px;     // search radius in pixels
  uint write_mask;     // union of character stencil write masks
  uint frame_id;
  uint mode;
  uint pad0;
  uint pad1;
  uint4 character_set[2];  // 256-bit set of character stencil values (uint4: cbuffer arrays pad each element to 16 bytes)
};

ConstantBuffer<TrackConstants> constants : register(b0, space0);
Texture2D<float> scene_depth : register(t0, space0);
Texture2D<uint2> scene_stencil : register(t1, space0);
RWByteAddressBuffer output : register(u0, space0);

// Output layout (uint offsets). Keep in sync with tracking.hpp.
static const uint kFrameId = 0;
static const uint kCharacterPixels = 1;
static const uint kHintStencil = 2;
static const uint kHintDepth = 3;
static const uint kBoxMinX = 4;  // pixel sums below are relative to box_min (avoids overflow)
static const uint kBoxMinY = 5;
static const uint kBuckets = 8;  // kSlotCount x kBucketCount x {count, sum_x, sum_y, max_depth}
static const uint kSlotCount = 16;
static const uint kBucketCount = 64;
static const uint kBucketFields = 4;
static const uint kHistogram = kBuckets + kSlotCount * kBucketCount * kBucketFields;  // 256 stencil counts
static const uint kTotal = kHistogram + 256;

uint SetWord(uint w) {
  return constants.character_set[w >> 2][w & 3];
}

// Rank of the value in the learned set (values above slot 15 share the last slot).
uint SlotOf(uint key) {
  uint word = key >> 5;
  uint slot = countbits(SetWord(word) & ((1u << (key & 31)) - 1u));
  for (uint w = 0; w < word; ++w) slot += countbits(SetWord(w));
  return min(slot, kSlotCount - 1);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (constants.mode == 0) {
    uint flat = id.y * 8 + id.x;  // one 8x8 group, loops over the buffer
    for (uint i = flat; i < kTotal; i += 64) {
      uint value = 0;
      if (i == kFrameId) value = constants.frame_id;
      if (i == kBoxMinX) value = asuint(constants.box_min.x);
      if (i == kBoxMinY) value = asuint(constants.box_min.y);
      output.Store(i * 4, value);
    }
    return;
  }

  int2 pixel = constants.box_min + int2(id.xy);
  if (any(pixel < 0) || any(pixel >= int2(constants.size))) return;
  float2 delta = float2(pixel) + 0.5f - constants.hint_px;
  float dist = length(delta);
  if (dist > constants.radius_px) return;

  uint stencil = scene_stencil.Load(int3(pixel, 0)).g;
  float depth = scene_depth.Load(int3(pixel, 0));

  if (all(pixel == int2(constants.hint_px))) {
    output.Store(kHintStencil * 4, stencil);
    output.Store(kHintDepth * 4, asuint(depth));
  }
  uint ignored;
  output.InterlockedAdd((kHistogram + stencil) * 4, 1, ignored);

  uint key = stencil & constants.write_mask;
  bool is_character = (SetWord(key >> 5) >> (key & 31)) & 1;
  if (!is_character || !(depth > 0.f)) return;

  // Reversed-Z: depth ~ near / distance, so log2(depth) bins are depth bands.
  uint bucket = (uint)clamp((log2(depth) + 16.f) * 4.f, 0.f, float(kBucketCount - 1));
  uint base = (kBuckets + (SlotOf(key) * kBucketCount + bucket) * kBucketFields) * 4;
  uint2 local = uint2(pixel - constants.box_min);
  output.InterlockedAdd(kCharacterPixels * 4, 1, ignored);
  output.InterlockedAdd(base + 0, 1, ignored);
  output.InterlockedAdd(base + 4, local.x, ignored);
  output.InterlockedAdd(base + 8, local.y, ignored);
  output.InterlockedMax(base + 12, asuint(depth), ignored);
}
