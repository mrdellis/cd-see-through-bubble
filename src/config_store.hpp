/*
 * See-Through Bubble: own settings file (SeeThroughBubble.ini next to the
 * add-on) instead of ReShade.ini. Plugged into RenoDX settings through
 * renodx::utils::settings::config_reader / config_writer.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

namespace config_store {

inline std::mutex mutex;
inline std::filesystem::path path;
inline std::map<std::string, std::map<std::string, std::string>> sections;

inline std::string Trim(const std::string& s) {
  const size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  return s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}

// Returns false if the file doesn't exist yet.
inline bool Load(const std::filesystem::path& file) {
  std::scoped_lock lock(mutex);
  path = file;
  sections.clear();
  std::ifstream in(file);
  if (!in) return false;
  std::string line, section;
  while (std::getline(in, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == ';' || line[0] == '#') continue;
    if (line.front() == '[' && line.back() == ']') {
      section = Trim(line.substr(1, line.size() - 2));
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    sections[section][Trim(line.substr(0, eq))] = Trim(line.substr(eq + 1));
  }
  return true;
}

inline void SaveLocked() {
  if (path.empty()) return;
  auto temporary = path;
  temporary += L".tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) return;
    out << "; See-Through Bubble settings (also editable in the ReShade overlay)\n";
    for (const auto& [section, values] : sections) {
      out << "\n[" << section << "]\n";
      for (const auto& [key, value] : values) out << key << '=' << value << '\n';
    }
  }
  std::error_code ec;
  std::filesystem::rename(temporary, path, ec);
}

inline bool Read(const std::string& section, const std::string& key, std::string& value) {
  std::scoped_lock lock(mutex);
  auto s = sections.find(section);
  if (s == sections.end()) return false;
  auto k = s->second.find(key);
  if (k == s->second.end()) return false;
  value = k->second;
  return true;
}

inline void Write(const std::string& section, const std::string& key, const std::string& value) {
  std::scoped_lock lock(mutex);
  auto& slot = sections[section][key];
  if (slot == value) return;
  slot = value;
  SaveLocked();
}

}  // namespace config_store
