#ifndef IO_H
#define IO_H

#include <string>
#include <iostream>
#include <filesystem>

namespace io{

void create_scratch(const std::string sdir, const bool debug=true);
void copy_scratch(const std::string sfrom, const std::string to, const bool debug=true);
void remove_scratch(const std::string sdir, const bool debug=true);
void remove_file(const std::string fname, const bool debug=true);
double directory_size(const std::filesystem::path& directory);
double available_disk();
void file_existQ(const std::string fname);

} // io

#endif
