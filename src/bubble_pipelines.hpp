/*
 * See-Through Bubble: pipeline cloning.
 *
 * Crimson Desert draws the world in two passes that both write depth:
 *   1. GBufferDepthPrePass: vsStandardDepthPre / vsTerrainMeshDepthPre with NO pixel shader.
 *   2. GBuffer: psStandard / psTerrainMesh with depth write + GreaterEqual.
 * We never edit the game's pipelines. For each matching pipeline we create a
 * clone at init time and bind it instead of the original while the bubble is on:
 *   - prepass clones get our discard-only pixel shader (bubble_prepass*.hlsl);
 *   - G-buffer clones get depth func Equal and no depth write, so pixels cut from
 *     the prepass fail the depth test and the occluder leaves no trace there.
 * Our pixel shader reads its settings from 20 root constants (b0, space73)
 * that we add to every D3D12 root signature after the game's root parameters
 * (before static samplers), so the game's root parameter indices, which its
 * ExecuteIndirect signatures rely on, stay the same.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <cstdint>
#include <shared_mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include <include/reshade.hpp>

#include "../../utils/hash.hpp"
#include "./dxc_runtime.hpp"
#include "./dxil.hpp"
#include "./ir_patch.hpp"
#include "./shared.h"
#include "./tracking.hpp"

namespace bubble {

using namespace reshade::api;

constexpr uint32_t kInjectionRegister = 0;   // must match shared.h
constexpr uint32_t kInjectionSpace = 73;
constexpr uint32_t kInjectionDwords = sizeof(ShaderInjectData) / 4;
constexpr uint32_t kMaxRootDwords = 64;
static_assert(sizeof(ShaderInjectData) % 16 == 0);

// Set by the add-on before Use().
inline ShaderInjectData* injection = nullptr;
// Prepass pixel shaders: the "exact" ones read the game's SceneConstantBuffer
// (b19, space35); the "basic" ones are the fallback if that pipeline fails to create.
inline std::span<const uint8_t> prepass_shader;
inline std::span<const uint8_t> prepass_terrain_shader;
inline std::span<const uint8_t> prepass_basic_shader;
inline std::span<const uint8_t> prepass_terrain_basic_shader;
inline std::atomic<uint32_t> stat_prepass_exact = 0;
inline std::atomic<uint32_t> stat_prepass_basic = 0;
// What to cut (0/1 at runtime, "See Through" settings) and the debug stencil filter.
inline const float* cut_vegetation = nullptr;      // kTree, kGrass
inline const float* cut_objects = nullptr;         // kGBuffer (psStandard) with any other stencil ref: buildings, fences
inline const float* cut_props = nullptr;           // kGBuffer (psStandard) with stencil ref kSmallObjectStencil: jars
inline const float* cut_terrain_meshes = nullptr;  // kPrepassTerrain, kGBufferTerrain (psTerrainMesh): rocks, cliffs
// psStandard draws for loose props (jars) use this stencil reference; buildings use 1
// (user test 2026-11-03; 5, 33, 36, 96, 133, 161, 224 cut nothing visible).
inline constexpr uint8_t kSmallObjectStencil = 129;
inline const float* stencil_only = nullptr;        // >= 0: cut only patched draws with this stencil reference
// bubble_snippet.ps_6_6.hlsl: source of the IR spliced into tree pixel shaders.
inline std::span<const uint8_t> snippet_shader;

enum class Kind : uint8_t { kPrepass, kPrepassTerrain, kGBuffer, kTree, kGrass, kGBufferTerrain };
constexpr int kKindCount = 6;

// Kinds whose clone runs an IR-patched copy of the game's pixel shader (the rest get our prepass PS).
inline bool IsPatchedKind(Kind kind) {
  return kind == Kind::kGBuffer || kind == Kind::kGBufferTerrain || kind == Kind::kTree || kind == Kind::kGrass;
}

struct Clone {
  pipeline clone = {0};
  pipeline_layout layout = {0};
  uint32_t injection_param = UINT32_MAX;
  Kind kind = Kind::kPrepass;
};

inline std::shared_mutex clones_mutex;
inline std::unordered_map<uint64_t, Clone> clones;

inline std::shared_mutex layouts_mutex;
inline std::unordered_map<uint64_t, uint32_t> injection_param_by_layout;

inline thread_local bool creating_clone = false;
inline thread_local std::vector<pipeline_layout_param> layout_params_storage;

inline std::atomic<uint32_t> stat_layouts_extended = 0;
inline std::atomic<uint32_t> stat_layouts_skipped = 0;
inline std::atomic<uint32_t> stat_clones[kKindCount] = {};
inline std::atomic<uint32_t> stat_clone_failures = 0;

struct __declspec(uuid("4f0b8d6e-5a57-4f39-9a51-3c1a2b7de1f4")) CmdState {
  const Clone* active = nullptr;  // clone bound in place of the game's pipeline, if any
  const Clone* bound = nullptr;   // clone of the bound pipeline, also when not swapped in
  pipeline original = {0};        // the game's bound pipeline (restored for draws the stencil filter skips)
  pipeline_stage active_stages = pipeline_stage::all;
  resource_view dsv = {0};              // bound depth-stencil view
  uint8_t character_write_mask = 0;     // non-zero while a character pipeline is bound
  uint64_t character_pipeline = 0;      // its handle (for the learn log)
  uint8_t stencil_reference = 0;        // last OMSetStencilRef
  bool scene_pass = false;              // a world G-buffer/tree pipeline is bound (its DSV is the scene depth)
  uint8_t marker = 0;                   // Marker of the bound pipeline (cutscene letterbox, minimap)
  float viewport_width = 0.f;
  float viewport_height = 0.f;
};

inline void Log(reshade::log::level lvl, const std::string& text) {
  reshade::log::message(lvl, ("seethroughbubble: " + text).c_str());
}

// ---------------------------------------------------------------- layouts

inline uint32_t RootDwordCost(uint32_t count, const pipeline_layout_param* params) {
  uint32_t cost = 0;
  for (uint32_t i = 0; i < count; ++i) {
    switch (params[i].type) {
      case pipeline_layout_param_type::push_constants:                cost += params[i].push_constants.count; break;
      case pipeline_layout_param_type::push_descriptors:              cost += 2; break;  // root descriptor
      case pipeline_layout_param_type::descriptor_table:
      case pipeline_layout_param_type::descriptor_table_with_static_samplers:
      case pipeline_layout_param_type::push_descriptors_with_ranges:  cost += 1; break;
      default:                                                        break;  // static samplers: free
    }
  }
  return cost;
}

inline bool IsStaticSamplerParam(const pipeline_layout_param& param) {
  return param.type == pipeline_layout_param_type::push_descriptors_with_static_samplers
         && param.descriptor_table_with_static_samplers.count != 0
         && param.descriptor_table_with_static_samplers.ranges[0].static_samplers != nullptr;
}

inline bool disable_layout_injection = false;  // bin64/seethroughbubble-disable-layouts exists
inline std::atomic<uint32_t> stat_layouts_initialized = 0;

inline std::string DescribeParams(uint32_t count, const pipeline_layout_param* params) {
  std::stringstream s;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& p = params[i];
    s << (i ? " " : "") << static_cast<int>(p.type);
    if (p.type == pipeline_layout_param_type::push_constants) {
      s << "(b" << p.push_constants.dx_register_index << ",s" << p.push_constants.dx_register_space << ",n"
        << p.push_constants.count << ",v" << static_cast<uint32_t>(p.push_constants.visibility) << ")";
    }
  }
  return s.str();
}

inline bool OnCreatePipelineLayout(device* device, uint32_t& param_count, pipeline_layout_param*& params) {
  if (disable_layout_injection) return false;
  if (device->get_api() != device_api::d3d12 || param_count == 0) return false;
  if (RootDwordCost(param_count, params) + kInjectionDwords > kMaxRootDwords) {
    stat_layouts_skipped++;
    return false;
  }
  // Insert before the first static-sampler param, as RenoDX does. ReShade keeps
  // static samplers last when converting to a D3D12 root signature (appending
  // after them crashed XeSS at startup). Static samplers aren't root parameters,
  // so the game's D3D12 root parameter indices don't move.
  uint32_t insert_index = param_count;
  for (uint32_t i = 0; i < param_count; ++i) {
    if (IsStaticSamplerParam(params[i])) {
      insert_index = i;
      break;
    }
  }
  constant_range range = {};
  // Same shape as RenoDX's proven injection: binding 0, visibility all. A
  // pixel-only param made every root signature fail to create (build 2): root
  // signatures that deny pixel-shader access can't hold a pixel-visible param.
  range.binding = 0;
  range.dx_register_index = kInjectionRegister;
  range.dx_register_space = kInjectionSpace;
  range.count = kInjectionDwords;
  range.visibility = shader_stage::all;
  layout_params_storage.assign(params, params + param_count);
  layout_params_storage.insert(layout_params_storage.begin() + insert_index, pipeline_layout_param(range));
  params = layout_params_storage.data();
  param_count = static_cast<uint32_t>(layout_params_storage.size());
  if (stat_layouts_extended++ < 6) {
    Log(reshade::log::level::info, "layout extended: " + DescribeParams(param_count, params) + " (insert at "
                                       + std::to_string(insert_index) + ")");
  }
  return true;
}

inline void OnInitPipelineLayout(device* device, uint32_t param_count, const pipeline_layout_param* params,
                                 pipeline_layout layout) {
  if (layout.handle == 0) return;
  for (uint32_t i = 0; i < param_count; ++i) {
    const auto& param = params[i];
    if (param.type != pipeline_layout_param_type::push_constants) continue;
    if (param.push_constants.dx_register_index != kInjectionRegister
        || param.push_constants.dx_register_space != kInjectionSpace) continue;
    if (stat_layouts_initialized++ < 3) {
      Log(reshade::log::level::info, "layout initialized with injection at param " + std::to_string(i));
    }
    std::unique_lock lock(layouts_mutex);
    injection_param_by_layout[layout.handle] = i;
    return;
  }
}

inline void OnDestroyPipelineLayout(device*, pipeline_layout layout) {
  std::unique_lock lock(layouts_mutex);
  injection_param_by_layout.erase(layout.handle);
}

// ---------------------------------------------------------------- pipelines

inline bool HasCode(const pipeline_subobject& sub) {
  const auto* desc = static_cast<const shader_desc*>(sub.data);
  return desc != nullptr && desc->code != nullptr && desc->code_size != 0;
}

inline std::string_view StageName(const pipeline_subobject& sub) {
  const auto* desc = static_cast<const shader_desc*>(sub.data);
  if (desc == nullptr || desc->code == nullptr || desc->code_size == 0) return {};
  return dxil::EntryPointName({static_cast<const uint8_t*>(desc->code), desc->code_size});
}

// ---------------------------------------------------------------- tree shader patching

inline std::mutex patch_mutex;  // guards the cache map only; DXC work runs in parallel
inline std::once_flag snippet_once;
inline std::optional<std::string> snippet_ir;  // disassembled once
inline std::string snippet_error;
inline uint32_t snippet_hash = 0;
inline std::filesystem::path cache_dir;  // set by the add-on; empty disables the disk cache
inline std::unordered_map<uint32_t, std::shared_ptr<const std::vector<uint8_t>>> patched_by_hash;  // null = failed
// Set when a patched shader shows that the game's SceneConstantBuffer layout no
// longer matches what our shaders read: then nothing is cut (prepass included).
inline std::atomic<bool> scene_layout_changed = false;
inline std::string scene_layout_detail;
inline std::atomic<uint32_t> stat_patched = 0;
inline std::atomic<uint32_t> stat_patched_from_disk = 0;
inline std::atomic<uint32_t> stat_patch_failures = 0;

inline std::optional<std::vector<uint8_t>> ReadCache(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return std::nullopt;
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (data.size() < 32 || std::memcmp(data.data(), "DXBC", 4) != 0) return std::nullopt;
  return data;
}

inline void WriteCache(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  auto temporary = path;
  temporary += L".tmp";
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file) return;
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }
  std::filesystem::rename(temporary, path, ec);
}

// Returns the game pixel shader with the bubble test spliced in, or null.
// Disk cache key: CRC32 of the original shader and of the snippet.
inline std::shared_ptr<const std::vector<uint8_t>> GetPatchedShader(std::span<const uint8_t> code, std::string_view name) {
  const uint32_t hash = renodx::utils::hash::ComputeCRC32(code.data(), code.size());
  {
    std::scoped_lock lock(patch_mutex);
    if (auto it = patched_by_hash.find(hash); it != patched_by_hash.end()) return it->second;
  }

  std::call_once(snippet_once, [] {
    snippet_hash = renodx::utils::hash::ComputeCRC32(snippet_shader.data(), snippet_shader.size());
    snippet_ir = dxc_runtime::Disassemble(snippet_shader, &snippet_error);
  });

  std::filesystem::path cache_file;
  if (!cache_dir.empty()) {
    char file_name[48];
    snprintf(file_name, sizeof(file_name), "%08X_%08X.cso", hash, snippet_hash);
    cache_file = cache_dir / file_name;
  }

  std::string error;
  std::shared_ptr<const std::vector<uint8_t>> result;
  std::string summary;
  if (!cache_file.empty()) {
    if (auto cached = ReadCache(cache_file)) {
      result = std::make_shared<const std::vector<uint8_t>>(std::move(*cached));
      stat_patched_from_disk++;
      summary = "disk cache";
    }
  }
  if (result == nullptr) {
    if (!snippet_ir.has_value()) {
      error = "snippet: " + snippet_error;
    } else if (auto target_ir = dxc_runtime::Disassemble(code, &error)) {
      if (auto patched = ir_patch::Patch(*target_ir, *snippet_ir, &error)) {
        if (auto container = dxc_runtime::AssembleAndValidate(patched->ir, &error)) {
          result = std::make_shared<const std::vector<uint8_t>>(std::move(*container));
          summary = patched->summary;
          if (!cache_file.empty()) WriteCache(cache_file, *result);
        }
      }
    }
  }
  if (result == nullptr && error.starts_with(ir_patch::kLayoutChangedError)) {
    if (!scene_layout_changed.exchange(true)) {
      scene_layout_detail = error;
      Log(reshade::log::level::error,
          "game's SceneConstantBuffer layout changed (" + error
              + "). The bubble is disabled until the add-on is updated.");
    }
  }
  if (result != nullptr) {
    if (stat_patched++ < 16) {
      std::stringstream s;
      s << "patched " << name << " 0x" << std::hex << hash << std::dec << " (" << summary << ")";
      Log(reshade::log::level::info, s.str());
    }
  } else if (stat_patch_failures++ < 12) {
    std::stringstream s;
    s << "patch failed for " << name << " 0x" << std::hex << hash << std::dec << ": " << error.substr(0, 600);
    Log(reshade::log::level::warning, s.str());
  }
  std::scoped_lock lock(patch_mutex);
  auto [it, inserted] = patched_by_hash.emplace(hash, result);
  return it->second;
}

inline bool IsTreePixelShader(std::string_view name) {
  return name.starts_with("PSMain8Bindless") && name.find("Grass") == std::string_view::npos;
}

inline bool IsGrassPixelShader(std::string_view name) {
  return (name.starts_with("PSMain8Bindless") || name.starts_with("GrassMain8Bindless"))
         && name.find("Grass") != std::string_view::npos;
}

// Game state markers, found by shader name (recon diff of gameplay vs cutscene):
//   ClearLetterBoxDepthPS: drawn every frame while cinematic letterbox bars are up (cutscenes);
//   PSMainMinimap: drawn every frame while the HUD minimap is visible.
enum Marker : uint8_t { kMarkerNone = 0, kMarkerLetterbox = 1, kMarkerMinimap = 2 };
inline std::atomic<uint64_t> present_count = 0;
inline std::atomic<uint64_t> last_marker_present[3] = {};
inline std::shared_mutex marker_mutex;
inline std::unordered_map<uint64_t, uint8_t> marker_pipelines;

// Presents since a marker pipeline last drew (UINT64_MAX if never).
inline uint64_t PresentsSinceMarker(Marker marker) {
  const uint64_t last = last_marker_present[marker].load(std::memory_order_relaxed);
  if (last == 0) return UINT64_MAX;
  return present_count.load(std::memory_order_relaxed) - last;
}

// Character pipelines (Character.hlsl) -> their stencil write mask, for tracking.
struct CharacterPipeline {
  uint8_t write_mask = 0;
  stencil_op pass_op = stencil_op::keep;
  compare_op stencil_func = compare_op::always;
  uint32_t rt_count = 0;
  std::string vs_name, ps_name;
};
inline std::shared_mutex character_mutex;
inline std::unordered_map<uint64_t, CharacterPipeline> character_pipelines;

// Logs which pipeline taught a new character stencil value (once per value).
inline void LogLearnedStencil(uint64_t pipeline_handle, uint8_t reference, uint8_t write_mask) {
  CharacterPipeline info;
  {
    std::shared_lock lock(character_mutex);
    if (auto it = character_pipelines.find(pipeline_handle); it != character_pipelines.end()) info = it->second;
  }
  std::stringstream s;
  s << "learned character stencil " << static_cast<int>(reference & write_mask) << " (ref " << static_cast<int>(reference)
    << ", write mask 0x" << std::hex << static_cast<int>(write_mask) << std::dec << ", pass op "
    << static_cast<int>(info.pass_op) << ", func " << static_cast<int>(info.stencil_func) << ", rts " << info.rt_count
    << ", vs " << info.vs_name << ", ps " << info.ps_name << ", pipeline 0x" << std::hex << pipeline_handle << ")";
  Log(reshade::log::level::info, s.str());
}

inline void OnInitPipeline(device* device, pipeline_layout layout, uint32_t subobject_count,
                           const pipeline_subobject* subobjects, pipeline pipeline) {
  if (creating_clone || pipeline.handle == 0 || device->get_api() != device_api::d3d12) return;

  std::string_view vs_name, ds_name, ps_name;
  int ps_index = -1, depth_index = -1;
  uint32_t rt_count = 0;
  format dsv_format = format::unknown;
  bool has_compute = false, has_mesh = false;
  for (uint32_t i = 0; i < subobject_count; ++i) {
    const auto& sub = subobjects[i];
    switch (sub.type) {
      case pipeline_subobject_type::vertex_shader:  if (sub.count) vs_name = StageName(sub); break;
      case pipeline_subobject_type::domain_shader:  if (sub.count) ds_name = StageName(sub); break;
      case pipeline_subobject_type::pixel_shader:
        ps_index = static_cast<int>(i);
        if (sub.count) ps_name = StageName(sub);
        break;
      case pipeline_subobject_type::compute_shader: if (sub.count) has_compute = HasCode(sub); break;
      case pipeline_subobject_type::mesh_shader:    if (sub.count) has_mesh = HasCode(sub); break;
      case pipeline_subobject_type::render_target_formats: rt_count = sub.count; break;
      case pipeline_subobject_type::depth_stencil_format:
        if (sub.count && sub.data) dsv_format = *static_cast<const format*>(sub.data);
        break;
      case pipeline_subobject_type::depth_stencil_state: if (sub.count && sub.data) depth_index = static_cast<int>(i); break;
      default: break;
    }
  }
  if (has_compute || has_mesh) return;

  depth_stencil_desc depth = {};  // D3D12 defaults when the subobject is absent
  if (depth_index >= 0) depth = *static_cast<const depth_stencil_desc*>(subobjects[depth_index].data);

  if (ps_name == "ClearLetterBoxDepthPS" || ps_name == "PSMainMinimap") {
    std::unique_lock lock(marker_mutex);
    marker_pipelines[pipeline.handle] = ps_name == "PSMainMinimap" ? kMarkerMinimap : kMarkerLetterbox;
  }

  // Character pipelines: depth prepass, hair, and the 4-target Equal G-buffer pass.
  const bool is_character = ps_name == "PSMainDepthPrepass" || ps_name == "PSMainHairStandard"
                            || (ps_name == "PSMain" && rt_count == 4 && depth.depth_func == compare_op::equal);
  if (is_character && depth.stencil_enable && depth.front_stencil_write_mask != 0) {
    std::unique_lock lock(character_mutex);
    character_pipelines[pipeline.handle] = {.write_mask = depth.front_stencil_write_mask,
                                            .pass_op = depth.front_stencil_pass_op,
                                            .stencil_func = depth.front_stencil_func,
                                            .rt_count = rt_count,
                                            .vs_name = std::string(vs_name),
                                            .ps_name = std::string(ps_name)};
  }

  std::optional<Kind> kind;
  const bool is_prepass_vs = vs_name == "vsStandardDepthPre" || vs_name == "vsTerrainMeshDepthPre"
                             || ds_name == "dsDeferredDepthPre";
  if (is_prepass_vs && ps_name.empty() && rt_count == 0 && dsv_format != format::unknown) {
    kind = vs_name == "vsTerrainMeshDepthPre" ? Kind::kPrepassTerrain : Kind::kPrepass;
  } else if ((ps_name == "psStandard" || ps_name == "psTerrainMesh") && rt_count == 5 && ps_index >= 0) {
    // World G-buffer: patched pixel shader (the old depth-Equal trick needed a
    // prepass, which the game doesn't always draw).
    kind = ps_name == "psTerrainMesh" ? Kind::kGBufferTerrain : Kind::kGBuffer;
  } else if (IsTreePixelShader(ps_name) && ps_index >= 0) {
    kind = Kind::kTree;
  } else if (IsGrassPixelShader(ps_name) && ps_index >= 0) {
    kind = Kind::kGrass;
  }
  if (!kind.has_value()) return;

  Clone clone = {.layout = layout, .kind = *kind};
  std::vector<pipeline_subobject> modified(subobjects, subobjects + subobject_count);
  shader_desc ps_desc = {};

  std::shared_ptr<const std::vector<uint8_t>> patched_ps;
  if (IsPatchedKind(*kind)) {
    {
      std::shared_lock lock(layouts_mutex);
      auto it = injection_param_by_layout.find(layout.handle);
      if (it == injection_param_by_layout.end()) {
        stat_clone_failures++;
        return;
      }
      clone.injection_param = it->second;
    }
    const auto* original = static_cast<const shader_desc*>(subobjects[ps_index].data);
    patched_ps = GetPatchedShader({static_cast<const uint8_t*>(original->code), original->code_size}, ps_name);
    if (patched_ps == nullptr) return;
    ps_desc = *original;
    ps_desc.code = patched_ps->data();
    ps_desc.code_size = patched_ps->size();
    modified[ps_index].data = &ps_desc;
  } else {
    {
      std::shared_lock lock(layouts_mutex);
      auto it = injection_param_by_layout.find(layout.handle);
      if (it == injection_param_by_layout.end()) {
        stat_clone_failures++;
        return;  // layout couldn't be extended; leave this pipeline alone
      }
      clone.injection_param = it->second;
    }
    ps_desc.entry_point = "main";
    if (ps_index >= 0) {
      modified[ps_index].count = 1;
      modified[ps_index].data = &ps_desc;
    } else {
      modified.push_back({pipeline_subobject_type::pixel_shader, 1, &ps_desc});
    }
  }

  const auto try_create = [&](std::span<const uint8_t> code) {
    if (*kind == Kind::kPrepass || *kind == Kind::kPrepassTerrain) {
      ps_desc.code = code.data();
      ps_desc.code_size = code.size();
    }
    creating_clone = true;
    const bool created =
        device->create_pipeline(layout, static_cast<uint32_t>(modified.size()), modified.data(), &clone.clone);
    creating_clone = false;
    return created && clone.clone.handle != 0;
  };
  bool ok = false;
  if (IsPatchedKind(*kind)) {
    ok = try_create({});
  } else {
    const bool terrain = *kind == Kind::kPrepassTerrain;
    ok = try_create(terrain ? prepass_terrain_shader : prepass_shader);
    if (ok) {
      stat_prepass_exact++;
    } else {
      ok = try_create(terrain ? prepass_terrain_basic_shader : prepass_basic_shader);
      if (ok) stat_prepass_basic++;
    }
  }
  if (!ok) {
    if (stat_clone_failures++ < 8) {
      std::stringstream s;
      s << "clone failed: kind " << static_cast<int>(*kind) << ", vs " << vs_name << ", ps " << ps_name;
      Log(reshade::log::level::warning, s.str());
    }
    return;
  }
  stat_clones[static_cast<int>(*kind)]++;
  std::unique_lock lock(clones_mutex);
  clones[pipeline.handle] = clone;
}

inline void OnDestroyPipeline(device* device, pipeline pipeline) {
  if (creating_clone) return;
  reshade::api::pipeline clone_handle = {0};
  {
    std::unique_lock lock(clones_mutex);
    auto it = clones.find(pipeline.handle);
    if (it == clones.end()) return;
    clone_handle = it->second.clone;
    clones.erase(it);
  }
  device->destroy_pipeline(clone_handle);
}

// ---------------------------------------------------------------- command lists

inline void OnInitCommandList(command_list* cmd_list) { cmd_list->create_private_data<CmdState>(); }
inline void OnDestroyCommandList(command_list* cmd_list) { cmd_list->destroy_private_data<CmdState>(); }
inline void OnResetCommandList(command_list* cmd_list) {
  if (auto* state = cmd_list->get_private_data<CmdState>()) *state = {};
}

inline bool Flag(const float* flag) { return flag == nullptr || *flag != 0.f; }

inline bool Enabled(Kind kind) {
  if (injection == nullptr || injection->bubble_enabled == 0.f) return false;
  if (scene_layout_changed.load(std::memory_order_relaxed)) return false;
  switch (kind) {
    case Kind::kTree:
    case Kind::kGrass:          return Flag(cut_vegetation);
    // psStandard: buildings and props share pipelines; PushIfNeeded picks per draw by
    // stencil.
    case Kind::kGBuffer:        return Flag(cut_objects) || Flag(cut_props);
    case Kind::kGBufferTerrain: return Flag(cut_terrain_meshes);
    // Depth prepasses: cut whenever any world category is on. Cutting the prepass of an
    // uncut object is harmless (its G-buffer draw fills the depth in again), and the
    // prepass kind doesn't reliably match the G-buffer kind (tessellated rocks use
    // dsDeferredDepthPre, classed kPrepass, but draw with psTerrainMesh).
    case Kind::kPrepass:
    case Kind::kPrepassTerrain: return Flag(cut_objects) || Flag(cut_props) || Flag(cut_terrain_meshes);
  }
  return true;
}

// psStandard G-buffer draw: cut it, given its stencil reference?
inline bool CutStandardDraw(uint8_t stencil_reference) {
  return Flag(stencil_reference == kSmallObjectStencil ? cut_props : cut_objects);
}

// Live counters, logged every 600 presents (see OnPresent).
inline std::atomic<uint64_t> n_binds_with_clone = 0;
inline std::atomic<uint64_t> n_swaps[kKindCount] = {};
inline std::atomic<uint64_t> n_skipped_disabled = 0;
inline std::atomic<uint64_t> n_no_state = 0;
inline std::atomic<uint64_t> n_pushes = 0;
inline std::atomic<uint64_t> n_draws_while_active = 0;
inline std::atomic<uint64_t> n_stencil_skipped = 0;
inline std::atomic<uint64_t> n_category_skipped = 0;  // psStandard draws left uncut by category (stencil)

// Stencil survey (log only): stencil reference per draw of each patched kind, to
// find out whether the stencil tells rocks from buildings.
inline std::atomic<uint32_t> stencil_survey[kKindCount][256] = {};

inline void OnBindPipeline(command_list* cmd_list, pipeline_stage stages, pipeline pipeline) {
  if (stages == pipeline_stage::compute_shader) return;
  auto* state = cmd_list->get_private_data<CmdState>();
  if (state != nullptr) {
    state->active = nullptr;
    state->bound = nullptr;
    state->original = pipeline;
    state->scene_pass = false;
    state->character_write_mask = 0;
    state->character_pipeline = 0;
    state->marker = kMarkerNone;
    {
      std::shared_lock lock(character_mutex);
      if (auto it = character_pipelines.find(pipeline.handle); it != character_pipelines.end()) {
        state->character_write_mask = it->second.write_mask;
        state->character_pipeline = pipeline.handle;
      }
    }
    std::shared_lock lock(marker_mutex);
    if (auto it = marker_pipelines.find(pipeline.handle); it != marker_pipelines.end()) state->marker = it->second;
  }

  const Clone* clone = nullptr;
  {
    std::shared_lock lock(clones_mutex);
    auto it = clones.find(pipeline.handle);
    if (it == clones.end()) return;
    clone = &it->second;  // stable: unordered_map nodes don't move; erased only on destroy
  }
  n_binds_with_clone++;
  if (state != nullptr) {
    state->bound = clone;
    state->scene_pass = clone->kind == Kind::kGBuffer || clone->kind == Kind::kGBufferTerrain || clone->kind == Kind::kTree;
  }
  if (!Enabled(clone->kind)) {
    n_skipped_disabled++;
    return;
  }
  cmd_list->bind_pipeline(stages, clone->clone);
  n_swaps[static_cast<int>(clone->kind)]++;
  if (state != nullptr) {
    state->active = clone;
    state->active_stages = stages;
  } else {
    n_no_state++;
  }
}

inline void OnBindViewports(command_list* cmd_list, uint32_t first, uint32_t count, const viewport* viewports) {
  if (first != 0 || count == 0) return;
  auto* state = cmd_list->get_private_data<CmdState>();
  if (state == nullptr) return;
  state->viewport_width = viewports[0].width;
  state->viewport_height = viewports[0].height;
}

inline void OnBindRenderTargets(command_list* cmd_list, uint32_t, const resource_view*, resource_view dsv) {
  if (auto* state = cmd_list->get_private_data<CmdState>()) state->dsv = dsv;
}

inline void OnBindPipelineStates(command_list* cmd_list, uint32_t count, const dynamic_state* states, const uint32_t* values) {
  auto* state = cmd_list->get_private_data<CmdState>();
  if (state == nullptr) return;
  for (uint32_t i = 0; i < count; ++i) {
    if (states[i] == dynamic_state::front_stencil_reference_value) state->stencil_reference = static_cast<uint8_t>(values[i]);
  }
}

inline void PushIfNeeded(command_list* cmd_list) {
  auto* state = cmd_list->get_private_data<CmdState>();
  if (state == nullptr) return;
  if (state->character_write_mask != 0) {
    if (tracking::RecordCharacterStencil(state->stencil_reference, state->character_write_mask)) {
      LogLearnedStencil(state->character_pipeline, state->stencil_reference, state->character_write_mask);
    }
  }
  if (state->scene_pass) tracking::NoteSceneDepthView(cmd_list->get_device(), state->dsv);
  if (state->marker != kMarkerNone) {
    last_marker_present[state->marker].store(present_count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }
#if SEETHROUGH_DEBUG
  if (state->bound != nullptr && IsPatchedKind(state->bound->kind)) {
    stencil_survey[static_cast<int>(state->bound->kind)][state->stencil_reference].fetch_add(1, std::memory_order_relaxed);
  }
#endif
  if (state->active == nullptr) return;
  const Clone* clone = state->active;
  // Debug filter: draws with another stencil reference run the game's own pipeline.
  if (stencil_only != nullptr && *stencil_only >= 0.f && IsPatchedKind(clone->kind)
      && state->stencil_reference != static_cast<uint8_t>(*stencil_only)) {
    cmd_list->bind_pipeline(state->active_stages, state->original);
    n_stencil_skipped++;
    return;
  }
  // Buildings vs small objects: same pipelines, told apart by the stencil reference.
  if (clone->kind == Kind::kGBuffer && !CutStandardDraw(state->stencil_reference)) {
    cmd_list->bind_pipeline(state->active_stages, state->original);
    n_category_skipped++;
    return;
  }
  n_draws_while_active++;
  // Re-bind right before the draw, as RenoDX's mods::shader does: a swap made in
  // the bind_pipeline event alone may not stick.
  cmd_list->bind_pipeline(state->active_stages, clone->clone);
  if (clone->injection_param == UINT32_MAX) return;
  ShaderInjectData data = *injection;
  if (state->viewport_width > 0.f) {
    data.render_width = state->viewport_width;
    data.render_height = state->viewport_height;
  }
  cmd_list->push_constants(shader_stage::pixel, clone->layout, clone->injection_param, 0, kInjectionDwords, &data);
  n_pushes++;
}

inline bool OnDraw(command_list* cmd_list, uint32_t, uint32_t, uint32_t, uint32_t) {
  PushIfNeeded(cmd_list);
  return false;
}
inline bool OnDrawIndexed(command_list* cmd_list, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
  PushIfNeeded(cmd_list);
  return false;
}
inline bool OnIndirect(command_list* cmd_list, indirect_command type, resource, uint64_t, uint32_t, uint32_t) {
  if (type == indirect_command::dispatch || type == indirect_command::dispatch_mesh
      || type == indirect_command::dispatch_rays) return false;
  PushIfNeeded(cmd_list);
  return false;
}

inline void OnPresent(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*) {
  present_count.fetch_add(1, std::memory_order_relaxed);
  static uint32_t frames = 0;
  if (++frames % 600 != 0) return;
  std::stringstream s;
  {
    s << "layouts extended " << stat_layouts_extended << " (initialized " << stat_layouts_initialized << ", skipped "
      << stat_layouts_skipped << "), clones: prepass " << stat_clones[0] << ", prepass terrain " << stat_clones[1]
      << ", gbuffer " << stat_clones[2] << ", gbuffer terrain " << stat_clones[5] << ", tree " << stat_clones[3] << ", grass " << stat_clones[4] << " (patched shaders " << stat_patched
      << ", from disk " << stat_patched_from_disk << ", patch failures " << stat_patch_failures << "), failures " << stat_clone_failures << ", prepass shader exact "
      << stat_prepass_exact << " / basic " << stat_prepass_basic << " | ";
  }
  s << "last 600 frames: binds with clone " << n_binds_with_clone.exchange(0) << ", swaps prepass "
    << n_swaps[0].exchange(0) << " / terrain " << n_swaps[1].exchange(0) << " / gbuffer " << n_swaps[2].exchange(0)
    << " / gbuffer terrain " << n_swaps[5].exchange(0) << " / tree " << n_swaps[3].exchange(0) << " / grass " << n_swaps[4].exchange(0)
    << ", stencil-filtered " << n_stencil_skipped.exchange(0) << ", category-skipped " << n_category_skipped.exchange(0)
    << ", skipped (disabled) " << n_skipped_disabled.exchange(0) << ", no cmd state " << n_no_state.exchange(0)
    << ", draws while active " << n_draws_while_active.exchange(0) << ", pushes " << n_pushes.exchange(0)
    << " | enabled " << (injection ? injection->bubble_enabled : -1.f) << ", objects "
    << (cut_objects ? *cut_objects : -1.f) << ", props " << (cut_props ? *cut_props : -1.f) << ", vegetation " << (cut_vegetation ? *cut_vegetation : -1.f) << ", terrain meshes "
    << (cut_terrain_meshes ? *cut_terrain_meshes : -1.f) << ", ignore depth "
    << (injection ? injection->debug_ignore_depth : -1.f) << ", radius " << (injection ? injection->bubble_radius : -1.f)
    << ", near " << (injection ? injection->camera_near : -1.f) << ", cut " << (injection ? injection->bubble_cut_distance : -1.f);
#if SEETHROUGH_DEBUG
  // Stencil survey: the most used stencil references per patched kind (value:draws).
  const struct { Kind kind; const char* name; } surveyed[] = {
      {Kind::kGBuffer, "standard"}, {Kind::kGBufferTerrain, "terrain mesh"}, {Kind::kTree, "tree"}};
  s << " | stencil refs";
  for (const auto& [kind, name] : surveyed) {
    std::vector<std::pair<uint32_t, int>> counts;
    for (int v = 0; v < 256; ++v) {
      const uint32_t n = stencil_survey[static_cast<int>(kind)][v].exchange(0, std::memory_order_relaxed);
      if (n != 0) counts.emplace_back(n, v);
    }
    std::sort(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    s << " " << name << ":";
    for (size_t i = 0; i < counts.size() && i < 10; ++i) s << " " << counts[i].second << "x" << counts[i].first;
  }
#endif
  Log(reshade::log::level::info, s.str());
}

inline void Use(DWORD fdw_reason) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      reshade::register_event<reshade::addon_event::create_pipeline_layout>(OnCreatePipelineLayout);
      reshade::register_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::register_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
      reshade::register_event<reshade::addon_event::destroy_pipeline>(OnDestroyPipeline);
      reshade::register_event<reshade::addon_event::init_command_list>(OnInitCommandList);
      reshade::register_event<reshade::addon_event::destroy_command_list>(OnDestroyCommandList);
      reshade::register_event<reshade::addon_event::reset_command_list>(OnResetCommandList);
      reshade::register_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
      reshade::register_event<reshade::addon_event::bind_viewports>(OnBindViewports);
      reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
      reshade::register_event<reshade::addon_event::bind_pipeline_states>(OnBindPipelineStates);
      reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(tracking::OnClearDepthStencil);
      reshade::register_event<reshade::addon_event::draw>(OnDraw);
      reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(OnIndirect);
      reshade::register_event<reshade::addon_event::present>(OnPresent);
      break;
    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::create_pipeline_layout>(OnCreatePipelineLayout);
      reshade::unregister_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::unregister_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline>(OnDestroyPipeline);
      reshade::unregister_event<reshade::addon_event::init_command_list>(OnInitCommandList);
      reshade::unregister_event<reshade::addon_event::destroy_command_list>(OnDestroyCommandList);
      reshade::unregister_event<reshade::addon_event::reset_command_list>(OnResetCommandList);
      reshade::unregister_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
      reshade::unregister_event<reshade::addon_event::bind_viewports>(OnBindViewports);
      reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
      reshade::unregister_event<reshade::addon_event::bind_pipeline_states>(OnBindPipelineStates);
      reshade::unregister_event<reshade::addon_event::clear_depth_stencil_view>(tracking::OnClearDepthStencil);
      reshade::unregister_event<reshade::addon_event::draw>(OnDraw);
      reshade::unregister_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(OnIndirect);
      reshade::unregister_event<reshade::addon_event::present>(OnPresent);
      break;
    default: break;
  }
}

}  // namespace bubble
