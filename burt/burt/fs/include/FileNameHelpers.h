/** @file
* C++ file name helpers
*/
#pragma once

#include <string_view>
#include <string>

namespace burt
{
    class FileNameHelpers
    {
    public:

        /** Is the path is relative to the current directory
        * @param path provided path
        * @return true if path starts with "./" or with ".\\"
        */
        static bool isRelativePath(std::string_view path);

        /** Build full filename via concatenating subPath and path with probably append pathSeparatorStr in between
        * @param subPath first path
        * @param path second part of the path
        * @param pathSeparatorStr c-string used as a separator string
        * @see normalizePath
        * @return constructed full string
        */
        static std::string buildFileName(const std::string& subPath, const std::string& path, char pathSeparator = '/');

        /** Perform path normalization by eliminating unnecessary things from it
        * @param path input constructed path or path that you have
        * @param pathSeparatorStr c-string used as a separator string
        * @return constructed normalized path
        */
        static std::string normalizePath(std::string_view path, char pathSeparator = '/');

        /* Change file extension for provided file path. And append extension if was not before
        * @param filepath base filename or full path for file contained backslashes or slashes
        * @param ext new file extension like ".zip" or "zip" or ".txt"
        * @return constructed file path with new extension
        * @remark can not work probably in the right way with file paths like /zzz/xxx/yy.ext1.ext2
        */
        static std::string changeFileExt(std::string_view filepath, std::string_view ext);

        /* Get file extension from file path provided in name
        * @param filepath base filename or full path for file contained backslashes or slashes
        * @return file extension like ".txt" or ".zip"
        */
        static std::string getFileExt(std::string_view filepath);

        /* Get file name without an extension from file path provided in the name
        * @param filepath base filename or full path for file contained backslashes or slashes
        * @return filepath without extension
        */
        static std::string cutFileExt(std::string_view filepath);

        /* Get file basename without folder paths
        * @param filepath base filename or full path for file contained backslashes or slashes
        * @return file basename, i.e. without directories part
        */
        static std::string extractBaseName(std::string_view filepath);

        /* Get folder name from file path
        * @param filepath base filename or full path for file contained backslashes or slashes
        * @return folder name where the file is stored, i.e. full directory name
        * @remark returned name of the folder can be an empty string if file path does not contain a folder
        */
        static std::string extractFolderName(std::string_view filePath);
    };
}
