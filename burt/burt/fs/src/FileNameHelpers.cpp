#include "FileNameHelpers.h"

#include "burt/fs/include/StringUtils.h"
#include "burt/system/include/PlatformSpecificMacroses.h"

#include <string>
#include <assert.h>
#include <stddef.h>

namespace 
{
    /**
     * Finds the position of the last occurrence of either a forward slash ("/") or a backslash ("\")
     * in the given string view.
     *
     * @param name The input string view in which to locate the last slash or backslash.
     * @return The position of the last occurrence of either a forward slash or a backslash.
     *         If neither is found, it returns std::string::npos.
     */
    size_t lastSlash(std::string_view name)
    {
        const size_t resSlash = name.rfind("/");
        const size_t resBackSlash = name.rfind("\\");

        if (resSlash != std::string::npos && resBackSlash == std::string::npos) {
            return resSlash;
        }
        else if (resSlash == std::string::npos && resBackSlash != std::string::npos) {
            return resBackSlash;
        } else {
            // return max of two values
            return (resSlash > resBackSlash ? resSlash : resBackSlash);
        }
    }

    /**
     * Finds the position of the last occurrence of a period (".") in the given string view.
     * Useful for determining file extensions.
     *
     * @param name The input string view in which to locate the last period.
     * @return The position of the last occurrence of a period, or std::string::npos if not found.
     */
    size_t lastPointForExension(std::string_view name) {
        return name.rfind(".");
    }
}

namespace burt
{
    bool FileNameHelpers::isRelativePath(std::string_view path)
    {
        if (path.length() == 0 || path.length() == 1)
            return false;
        else
        {
            return (path[0] == '.' && path[1] == '/') || (path[0] == '.' && path[1] == '\\');
        }
    }

    std::string FileNameHelpers::buildFileName(const std::string& subPath, const std::string& path, char pathSeparator)
    {
        if (subPath.length() == 0)
            return path;

        size_t pos = subPath.length() - 1;

        std::string	res(subPath);

        if (subPath[pos] == '\\' || subPath[pos] == '/' || subPath[pos] == pathSeparator)
        {
            res += path;
        }
        else
        {
            res += pathSeparator;
            res += path;
        }
        return res;
    }

    std::string	FileNameHelpers::normalizePath(std::string_view path, char pathSeparator)
    {
        std::string	res(path);

        for (size_t i = 0; i < res.length(); ++i)
        {
            if (res[i] == '\\' || res[i] == '/')
                res[i] = pathSeparator;
        }

        std::string pathSepStr = "";
        pathSepStr += pathSeparator;
        
        std::string pattern[] = {
            std::string(pathSepStr) + "." + std::string(pathSepStr),
            std::string(pathSepStr) + std::string(pathSepStr)
        };

        for (size_t j = 0; j < sizeof(pattern)/sizeof(pattern[0]); ++j)
        {
            for (;;)
            {
                size_t findIndex = res.find(pattern[j]);
                if (findIndex == std::string::npos)
                    break;

                res.replace(findIndex, pattern[j].length(), pathSepStr);
            }
        }

        while (!res.empty() && res.back() == pathSeparator)
        {
            res.erase(res.begin() + res.length() - 1);
        }

        return res;
    }


    std::string	FileNameHelpers::changeFileExt(std::string_view filepath, std::string_view ext)
    {
        if (ext.length() == 0)
        {
            assert(!"You're passed empty extension string to change extension of");
        }

        std::string usedExt;

        if (ext.front() != '.' )
        {
            usedExt = std::string(".") + std::string(ext);
        }
        else
        {
            usedExt = ext;
        }

        size_t foundLastPoint = lastPointForExension(filepath);
        size_t foundLastSlash = lastSlash(filepath);

        if (foundLastPoint == std::string_view::npos)
        {
            return burt::string_utils::concat(filepath, usedExt);
        }
        else
        {
            if (foundLastSlash != std::string::npos)
            {
                // The last point in part of path before last slash, so point is memmber of name of some folder
                if (foundLastPoint < foundLastSlash) 
                {
                    return burt::string_utils::concat(filepath, usedExt);
                }
                else
                {
                    return burt::string_utils::concat(filepath.substr(0, foundLastPoint), usedExt);
                }
            }
            else
            {
                // The path contained last point
                return burt::string_utils::concat(filepath.substr(0, foundLastPoint), usedExt);
            }
        }
    }

    std::string FileNameHelpers::getFileExt(std::string_view filename)
    {
        size_t foundLastPoint = lastPointForExension(filename);
        size_t foundLastSlash = lastSlash(filename);

        if (foundLastPoint == std::string_view::npos)
        {
            return std::string();
        }

        if (foundLastSlash != std::string_view::npos)
        {
            if (foundLastPoint < foundLastSlash)
            {
                return std::string();
            }
        }

        return std::string(filename.substr(foundLastPoint));
    }

    std::string FileNameHelpers::cutFileExt(std::string_view filepath)
    {
        size_t foundLastPoint = lastPointForExension(filepath);
        if (foundLastPoint == std::string_view::npos)
        {
            return std::string(filepath);
        }
        return std::string(filepath.substr(0, foundLastPoint));
    }

    std::string FileNameHelpers::extractBaseName(std::string_view filepath)
    {
        size_t pos = lastSlash(filepath);
        if (pos == std::string::npos)
        {
            return std::string(filepath);
        }
        else
        {
            //std::string name = std::string(filepath);
            //name.erase(0, pos + 1);

            std::string name(filepath.substr(pos + 1));
            return name;
        }
    }

    std::string FileNameHelpers::extractFolderName(std::string_view filePath)
    {
        size_t pos = lastSlash(filePath);

        if (pos == std::string_view::npos)
        {
            return std::string();
        }
        else
        {
            std::string name(filePath.substr(0, pos + 1));
            return name;
        }
    }
}
