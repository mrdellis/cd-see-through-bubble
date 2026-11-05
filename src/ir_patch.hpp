/*
 * See-Through Bubble: LLVM IR text patcher.
 *
 * Splices the bubble test (compiled from bubble_snippet.ps_6_6.hlsl and
 * disassembled at runtime) into the entry function of a game pixel shader's
 * disassembly, so the game's own shading is kept and a discard is added:
 *   - the snippet's values are renamed %N -> %bub_N (the game's numbering stays valid);
 *   - its SceneConstantBuffer binding is remapped to the one the target declares;
 *   - its SV_Position input id is remapped to the target's;
 *   - our cbuffer (b0, space73) is appended to the target's CBV metadata;
 *   - missing dx.op declarations are added with their attribute groups;
 *   - ForceEarlyDepthStencil is cleared (otherwise discard keeps the depth write).
 * Plain string handling only: MSVC std::regex recursion overflows on long IR lines.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "./shared.h"

namespace ir_patch {

namespace internal {

inline std::vector<std::string> SplitLines(std::string_view text) {
  std::vector<std::string> lines;
  size_t start = 0;
  while (start <= text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string_view::npos) end = text.size();
    std::string_view line = text.substr(start, end - start);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    lines.emplace_back(line);
    if (end == text.size()) break;
    start = end + 1;
  }
  return lines;
}

inline bool StartsWith(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }

inline std::optional<int64_t> ParseInt(std::string_view s) {
  int64_t value = 0;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec != std::errc() || ptr == s.data()) return std::nullopt;
  return value;
}

// "!123 = ..." -> 123
inline std::optional<int64_t> MetadataId(std::string_view line) {
  if (line.size() < 2 || line[0] != '!' || line[1] < '0' || line[1] > '9') return std::nullopt;
  size_t end = line.find(" = ");
  if (end == std::string_view::npos) return std::nullopt;
  return ParseInt(line.substr(1, end - 1));
}

// "!123 = !{a, b, c}" -> {"a","b","c"}
inline std::optional<std::vector<std::string>> MetadataTuple(std::string_view line) {
  size_t open = line.find(" = !{");
  if (open == std::string_view::npos || line.back() != '}') return std::nullopt;
  std::string_view body = line.substr(open + 5, line.size() - (open + 5) - 1);
  std::vector<std::string> items;
  size_t start = 0;
  while (start <= body.size()) {
    size_t comma = body.find(", ", start);
    if (comma == std::string_view::npos) comma = body.size();
    items.emplace_back(body.substr(start, comma - start));
    if (comma == body.size()) break;
    start = comma + 2;
  }
  return items;
}

inline std::string JoinTuple(int64_t id, const std::vector<std::string>& items) {
  std::string s = "!" + std::to_string(id) + " = !{";
  for (size_t i = 0; i < items.size(); ++i) s += (i ? ", " : "") + items[i];
  return s + "}";
}

inline int FindLine(const std::vector<std::string>& lines, std::string_view prefix) {
  for (size_t i = 0; i < lines.size(); ++i) {
    if (StartsWith(lines[i], prefix)) return static_cast<int>(i);
  }
  return -1;
}

// Name of a declared/defined function: text between '@' and '('.
inline std::string FunctionName(std::string_view line) {
  size_t at = line.find('@');
  size_t paren = line.find('(', at);
  if (at == std::string_view::npos || paren == std::string_view::npos) return {};
  return std::string(line.substr(at + 1, paren - at - 1));
}

// Trailing "#N" attribute group of a declaration.
inline std::optional<int64_t> AttributeGroup(std::string_view line) {
  size_t hash = line.rfind(" #");
  if (hash == std::string_view::npos) return std::nullopt;
  return ParseInt(line.substr(hash + 2));
}

// %123 -> %bub_123 (numbered values only; named values like %dx.types.X untouched).
inline std::string RenameNumberedValues(std::string_view text) {
  std::string out;
  out.reserve(text.size() + text.size() / 8);
  for (size_t i = 0; i < text.size(); ++i) {
    out.push_back(text[i]);
    if (text[i] != '%' || i + 1 >= text.size() || text[i + 1] < '0' || text[i + 1] > '9') continue;
    size_t j = i + 1;
    while (j < text.size() && text[j] >= '0' && text[j] <= '9') ++j;
    if (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_' || text[j] == '.')) continue;
    out += "bub_";
  }
  return out;
}

inline bool ReplaceOnce(std::string* text, std::string_view from, std::string_view to) {
  size_t pos = text->find(from);
  if (pos == std::string::npos) return false;
  text->replace(pos, from.size(), to);
  return true;
}

inline size_t ReplaceAll(std::string* text, std::string_view from, std::string_view to) {
  size_t count = 0;
  for (size_t pos = text->find(from); pos != std::string::npos; pos = text->find(from, pos + to.size())) {
    text->replace(pos, from.size(), to);
    ++count;
  }
  return count;
}

}  // namespace internal

struct Result {
  std::string ir;
  std::string summary;
};

// Error prefix when the game's SceneConstantBuffer no longer has the fields our
// bubble code reads at the offsets it expects (after a game patch).
inline constexpr std::string_view kLayoutChangedError = "scene layout changed";

// Field offsets our shaders hard-code (bubble.hlsli / bubble_snippet: packoffset c4 and c30).
struct ExpectedField {
  std::string_view name;
  int offset;
};
inline constexpr ExpectedField kExpectedSceneFields[] = {
    {"_bufferSizeAndInvSize", 64},
    {"_viewPos", 128},
    {"_viewProjRelative", 288},
    {"_invViewProjRelative", 480},
};

enum class LayoutCheck { kOk, kChanged, kUnverifiable };

// Checks the reflection comment block of the disassembly:
//   ; cbuffer __3__35__0__0__SceneConstantBuffer ... ;       float4 _bufferSizeAndInvSize;   ; Offset:   64
inline LayoutCheck CheckSceneLayout(const std::vector<std::string>& lines, std::string* detail) {
  int start = -1;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i].find("; cbuffer ") == 0 && lines[i].find("SceneConstantBuffer") != std::string::npos) {
      start = static_cast<int>(i);
      break;
    }
  }
  if (start < 0) return LayoutCheck::kUnverifiable;
  for (const auto& field : kExpectedSceneFields) {
    int found_offset = -1;
    for (size_t i = start; i < lines.size() && i < static_cast<size_t>(start) + 400; ++i) {
      const auto& line = lines[i];
      if (i > static_cast<size_t>(start) && line.find("; cbuffer ") == 0) break;  // next cbuffer
      const size_t name_pos = line.find(std::string(" ") + std::string(field.name) + ";");
      if (name_pos == std::string::npos) continue;
      const size_t offset_pos = line.find("Offset:", name_pos);
      if (offset_pos == std::string::npos) continue;
      found_offset = std::atoi(line.c_str() + offset_pos + 7);
      break;
    }
    if (found_offset != field.offset) {
      if (detail != nullptr) {
        *detail = std::string(field.name) + " at " + (found_offset < 0 ? std::string("(missing)") : std::to_string(found_offset))
                  + ", expected " + std::to_string(field.offset);
      }
      return LayoutCheck::kChanged;
    }
  }
  return LayoutCheck::kOk;
}

// Snippet facts (from bubble_snippet.ps_6_6.hlsl): SceneConstantBuffer at b19,
// space35 with a 544-byte declared range; SV_Position is input 0.
inline constexpr std::string_view kSnippetSceneBind = "%dx.types.ResBind { i32 19, i32 19, i32 35, i8 2 }, i32 19,";
inline constexpr std::string_view kSnippetSceneProps = "%dx.types.ResourceProperties { i32 13, i32 544 }";
inline constexpr std::string_view kSnippetLoadPosition = "@dx.op.loadInput.f32(i32 4, i32 0,";

inline std::optional<Result> Patch(std::string_view target_text, std::string_view snippet_text, std::string* error) {
  using namespace internal;
  auto fail = [&](std::string message) -> std::optional<Result> {
    if (error != nullptr) *error = std::move(message);
    return std::nullopt;
  };

  auto target = SplitLines(target_text);
  auto snippet = SplitLines(snippet_text);

  std::string layout_detail;
  const LayoutCheck layout = CheckSceneLayout(target, &layout_detail);
  if (layout == LayoutCheck::kChanged) return fail(std::string(kLayoutChangedError) + ": " + layout_detail);

  // --- target: SceneConstantBuffer CBV (space, register, size)
  int64_t scene_space = -1, scene_register = -1, scene_size = -1;
  for (const auto& line : target) {
    if (!MetadataId(line).has_value()) continue;
    size_t marker = line.find("SceneConstantBuffer* undef, !\"\", ");
    if (marker == std::string::npos) continue;
    auto items = MetadataTuple(line);
    if (!items || items->size() < 7) continue;
    auto space = ParseInt(std::string_view((*items)[3]).substr(4));
    auto reg = ParseInt(std::string_view((*items)[4]).substr(4));
    auto size = ParseInt(std::string_view((*items)[6]).substr(4));
    if (!space || !reg || !size) continue;
    scene_space = *space, scene_register = *reg, scene_size = *size;
    break;
  }
  if (scene_space < 0) return fail("target has no SceneConstantBuffer CBV");

  // --- target: SV_Position input element id
  int64_t sv_position_id = -1;
  for (const auto& line : target) {
    if (!MetadataId(line).has_value()) continue;
    size_t marker = line.find(", !\"SV_Position\"");
    if (marker == std::string::npos || line.find(" = !{i32 ") == std::string::npos) continue;
    auto items = MetadataTuple(line);
    if (!items || items->empty()) continue;
    auto id = ParseInt(std::string_view((*items)[0]).substr(4));
    if (id) sv_position_id = *id;
    break;
  }
  if (sv_position_id < 0) return fail("target has no SV_Position input");

  // --- target: metadata ids, resource lists
  int64_t next_md = 0;
  for (const auto& line : target) {
    if (auto id = MetadataId(line)) next_md = (std::max)(next_md, *id + 1);
  }
  int res_line = FindLine(target, "!dx.resources = !{!");
  if (res_line < 0) return fail("target has no !dx.resources");
  auto res_ref = target[res_line].substr(std::string_view("!dx.resources = !{!").size());
  res_ref.pop_back();  // '}'
  int res_node = FindLine(target, "!" + res_ref + " = !{");
  if (res_node < 0) return fail("!dx.resources node missing");
  auto res_items = MetadataTuple(target[res_node]);
  if (!res_items || res_items->size() != 4) return fail("unexpected !dx.resources shape");

  std::vector<std::string> new_metadata;
  const int64_t our_cbv_md = next_md++;
  int64_t cbv_count = 0;
  if ((*res_items)[2] == "null") {
    const int64_t list_md = next_md++;
    new_metadata.push_back("!" + std::to_string(list_md) + " = !{!" + std::to_string(our_cbv_md) + "}");
    (*res_items)[2] = "!" + std::to_string(list_md);
    target[res_node] = JoinTuple(std::stoll(res_ref), *res_items);
  } else {
    int list_line = FindLine(target, (*res_items)[2] + " = !{");
    if (list_line < 0) return fail("CBV list node missing");
    auto list_items = MetadataTuple(target[list_line]);
    if (!list_items) return fail("bad CBV list node");
    cbv_count = static_cast<int64_t>(list_items->size());
    list_items->push_back("!" + std::to_string(our_cbv_md));
    target[list_line] = JoinTuple(*MetadataId(target[list_line]), *list_items);
  }
  new_metadata.insert(new_metadata.begin(),
                      "!" + std::to_string(our_cbv_md) + " = !{i32 " + std::to_string(cbv_count)
                          + ", %bub_shader_injection* undef, !\"\", i32 73, i32 0, i32 1, i32 "
                          + std::to_string(sizeof(ShaderInjectData)) + ", null}");

  // --- target: clear ForceEarlyDepthStencil (shader flags bit 3) on the entry
  bool cleared_early_depth = false;
  if (int entry_line = FindLine(target, "!dx.entryPoints = !{!"); entry_line >= 0) {
    auto entry_ref = target[entry_line].substr(std::string_view("!dx.entryPoints = !{!").size());
    entry_ref.pop_back();
    if (int entry_node = FindLine(target, "!" + entry_ref + " = !{"); entry_node >= 0) {
      auto entry_items = MetadataTuple(target[entry_node]);
      if (entry_items && entry_items->size() >= 5 && (*entry_items)[4] != "null") {
        int props_line = FindLine(target, (*entry_items)[4] + " = !{i32 0, i64 ");
        if (props_line >= 0) {
          auto props = MetadataTuple(target[props_line]);
          if (props && props->size() >= 2) {
            auto flags = ParseInt(std::string_view((*props)[1]).substr(4));
            if (flags && (*flags & 8) != 0) {
              (*props)[1] = "i64 " + std::to_string(*flags & ~int64_t{8});
              target[props_line] = JoinTuple(*MetadataId(target[props_line]), *props);
              cleared_early_depth = true;
            }
          }
        }
      }
    }
  }

  // --- snippet body (straight-line code up to the final conditional branch)
  int snippet_define = FindLine(snippet, "define void @");
  if (snippet_define < 0) return fail("snippet has no entry");
  std::string body;
  std::string condition;
  for (size_t i = snippet_define + 1; i < snippet.size(); ++i) {
    std::string_view line = snippet[i];
    if (StartsWith(line, "  br i1 %")) {
      size_t start = std::string_view("  br i1 ").size();
      condition = std::string(line.substr(start, line.find(',', start) - start));
      break;
    }
    if (line.empty() || StartsWith(line, "; <label>") || StartsWith(line, "}")) return fail("snippet is not straight-line");
    if (size_t comment = line.find("  ; "); comment != std::string_view::npos) line = line.substr(0, comment);
    body.append(line).push_back('\n');
  }
  if (condition.empty()) return fail("snippet has no final branch");
  body += "  call void @dx.op.discard(i32 82, i1 " + condition + ")";
  body = RenameNumberedValues(body);

  const std::string scene_bind = "%dx.types.ResBind { i32 " + std::to_string(scene_register) + ", i32 "
                                 + std::to_string(scene_register) + ", i32 " + std::to_string(scene_space)
                                 + ", i8 2 }, i32 " + std::to_string(scene_register) + ",";
  const std::string scene_props = "%dx.types.ResourceProperties { i32 13, i32 " + std::to_string(scene_size) + " }";
  if (!ReplaceOnce(&body, kSnippetSceneBind, scene_bind)) return fail("snippet scene binding not found");
  if (!ReplaceOnce(&body, kSnippetSceneProps, scene_props)) return fail("snippet scene properties not found");
  if (ReplaceAll(&body, kSnippetLoadPosition,
                 "@dx.op.loadInput.f32(i32 4, i32 " + std::to_string(sv_position_id) + ",")
      == 0) {
    return fail("snippet SV_Position load not found");
  }

  // --- declarations and attribute groups
  std::set<std::string> target_functions;
  int64_t next_attribute = 0;
  for (const auto& line : target) {
    if (StartsWith(line, "declare ")) target_functions.insert(FunctionName(line));
    if (StartsWith(line, "attributes #")) {
      if (auto id = ParseInt(std::string_view(line).substr(12))) next_attribute = (std::max)(next_attribute, *id + 1);
    }
  }
  std::vector<std::pair<int64_t, int64_t>> attribute_map;  // snippet group -> new group
  std::vector<std::string> new_declarations, new_attributes;
  for (const auto& line : snippet) {
    if (!StartsWith(line, "declare ")) continue;
    if (target_functions.contains(FunctionName(line))) continue;
    auto group = AttributeGroup(line);
    if (!group) return fail("snippet declaration without attributes");
    int64_t mapped = -1;
    for (auto [from, to] : attribute_map) {
      if (from == *group) mapped = to;
    }
    if (mapped < 0) {
      int attr_line = FindLine(snippet, "attributes #" + std::to_string(*group) + " = ");
      if (attr_line < 0) return fail("snippet attribute group missing");
      mapped = next_attribute++;
      attribute_map.emplace_back(*group, mapped);
      std::string_view attr = snippet[attr_line];
      new_attributes.push_back("attributes #" + std::to_string(mapped) + std::string(attr.substr(attr.find(" = "))));
    }
    std::string_view decl = line;
    new_declarations.push_back(std::string(decl.substr(0, decl.rfind(" #"))) + " #" + std::to_string(mapped));
  }

  // --- types
  std::string inject_type = "%bub_struct.ShaderInjectData = type { float";
  for (size_t i = 1; i < sizeof(ShaderInjectData) / 4; ++i) inject_type += ", float";
  inject_type += " }";
  std::vector<std::string> new_types = {
      inject_type,
      "%bub_shader_injection = type { %bub_struct.ShaderInjectData }",
  };
  for (std::string_view needed : {"%dx.types.Handle = type", "%dx.types.ResBind = type",
                                  "%dx.types.ResourceProperties = type", "%dx.types.CBufRet.f32 = type"}) {
    if (FindLine(target, needed) >= 0) continue;
    int line = FindLine(snippet, needed);
    if (line < 0) return fail("snippet type missing");
    new_types.push_back(snippet[line]);
  }

  // --- emit
  int first_type = -1, target_define = -1, first_attribute = -1;
  for (size_t i = 0; i < target.size(); ++i) {
    if (first_type < 0 && StartsWith(target[i], "%")) first_type = static_cast<int>(i);
    if (target_define < 0 && StartsWith(target[i], "define void @")) target_define = static_cast<int>(i);
    if (first_attribute < 0 && StartsWith(target[i], "attributes #")) first_attribute = static_cast<int>(i);
  }
  if (first_type < 0 || target_define < 0) return fail("target layout not recognized");

  Result result;
  std::string& out = result.ir;
  out.reserve(target_text.size() + body.size() + 4096);
  for (size_t i = 0; i < target.size(); ++i) {
    if (static_cast<int>(i) == first_type) {
      for (const auto& t : new_types) out.append(t).push_back('\n');
    }
    if (static_cast<int>(i) == first_attribute) {
      for (const auto& d : new_declarations) out.append(d).push_back('\n');
    }
    out.append(target[i]).push_back('\n');
    if (static_cast<int>(i) == target_define) out.append(body).push_back('\n');
  }
  if (first_attribute < 0) {
    for (const auto& d : new_declarations) out.append(d).push_back('\n');
  }
  for (const auto& a : new_attributes) out.append(a).push_back('\n');
  for (const auto& m : new_metadata) out.append(m).push_back('\n');

  result.summary = "scene b" + std::to_string(scene_register) + " space" + std::to_string(scene_space) + ", SV_Position "
                   + std::to_string(sv_position_id) + ", cbv " + std::to_string(cbv_count) + ", +"
                   + std::to_string(new_declarations.size()) + " decls"
                   + (cleared_early_depth ? ", cleared early depth" : "")
                   + (layout == LayoutCheck::kUnverifiable ? ", layout unverified" : "");
  return result;
}

}  // namespace ir_patch
