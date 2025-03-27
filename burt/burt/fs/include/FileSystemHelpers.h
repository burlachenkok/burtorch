/** @file
* C++ file system helpers
*/

#pragma once

#include <string>
#include <stdint.h>

#if BURT_WINDOWS
    #include <sys/stat.h>
    #include <direct.h>
    #include <io.h>

#elif BURT_LINUX || BURT_MACOS
    #include <sys/stat.h>
    #include <unistd.h>
    // Unix doesn't differentiate between binary and text files
    #ifndef O_BINARY
    #define O_BINARY 0
    #endif
#else
    // Another includes
#endif

#include <fcntl.h>

//=================================================================================//
/* Read user permission */
#if !defined(S_IRUSR)
    #define S_IRUSR S_IREAD
#endif

/* Write user permission */
#if !defined(S_IWUSR)
    #define S_IWUSR S_IWRITE
#endif

/* Read group permission */
#if !defined(S_IRGRP)
    #define S_IRGRP 0
#endif

/* Write group permission */
#if !defined(S_IWGRP)
    #define S_IWGRP 0
#endif

/* Read others permission */
#if !defined(S_IROTH)
    #define S_IROTH 0
#endif

/* Write others permission */
#if !defined(S_IWOTH)
    #define S_IWOTH 0
#endif
//=================================================================================//

namespace burt
{
    class FileSystemHelpers
    {
    public:
        /** Get the current working directory
        *@return current working directory path
        */
        static std::string  getCwd();

        /** Change the current working directory
        *@param path path for new directory
        *@return true if changing directory has succeeded
        */
        static bool chDir(const std::string& path);

        /** Does file exist
        *@param path filename
        *@return returns true if files exist
        *@remark what is check that file is a regular file
        */
        static bool isFileExist(const std::string& path);

        /** Does file or directory exist
        *@param path full filename of file/directory
        *@return true if file or directory exists
        */
        static bool isFileOrFolderExist(const std::string& path);

        /** Does directory exist
        *@param path path to the directory
        *@return return true if directory exists
        */
        static bool isDirExist(const std::string& path);

        /** Get the file size in bytes
        *@param path path for the file
        *@return size in bytes.
         * @remark If the file does not exist or is empty, the function returns 0.
        */
        static uint64_t getFileSize(const std::string& path);

        /** Get a number of nonempty lines in a text file. Line delimiter is <CR LF> or <LF>. So this method can be used for text files in Windows, Linux, and Mac OS.
        *@param path path for the file
        *@return size in bytes. If the file does not exist or is empty, the function returns 0.
        */
        static uint32_t nonEmptyLinesInFile(const std::string& path);

        /** Create directory
        *@param path directory name
        *@return true if the directory has been created successfully
        */
        static bool createDir(const std::string& path);

        /** Remove directory
        *@param path directory name
        *@return true if the directory has been removed successfully
        */
        static bool removeDir(const std::string& path);

        /** Remove file
        *@param path name of the file
        *@return true if the file has been removed successfully
        */
        static bool removeFile(const std::string& path);

        /** Save content to file
        * @param fileName name of the file
        * @param rawBuffer pointer to the first byte of the raw buffer
        * @param rawBufferSize raw size of the buffer
        * @return true if the file has been removed successfully
        */
        static bool saveFile(const std::string& fileName, void* rawBuffer, size_t rawBufferSize);

        struct FileMappingResult
        {
            void* memory;               ///< Mapped memory from view of file
            uint64_t memorySizeInBytes; ///< Memory size in bytes
            uint64_t fileSizeInBytes;   ///< Size of file in bytes
            const char* errorMsg;       ///< Error message

            bool isReadOnly;            ///< Used memory view should be used for read-only
            bool isOk;                  ///< Memory view is valid
        };

        /** Create a view of all the content of the file by mapping it into the virtual address space of the process
        * @param fname name of file
        * @param isReadOnly open file in read-only mode
        * @param isCreareIfNotExist Create the file if it does not exist in the filesystem
        * @return The FileMappingResult structure with detailed information
        * @deprecated optUseCaching - use filesystem caching, optSequntial - Access provides a hint that it is most likely file will be accessed in a sequential manner
        */
        static FileMappingResult mapFileToMemory(const char* fname, bool isReadOnly, bool isCreareIfNotExist = false);

        static FileMappingResult mapFileToMemoryForWrite(const char* fname, uint32_t file_size);
                
        /** Unmap previously mapped content of the file via mapFileToMemory
        * @param viewOfFile structure that contains information about mapped content of the file  of file
        * @return true if content has been unmapped
        * @remark Internally viewOfFile.memory is reset in the passed reference. It is safe to unmap several times.
        */
        static bool unmapFileFromMemory(FileMappingResult& viewOfFile);

        /** Due to performance reasons OS may not immediately update the file on the disk in which the memory map is opened.
        * Call this function to force OS to flush all changes to disk.
        * @param viewOfFile structure that contains information about mapped content of the file of file.
        * @return true if all changes have been flushed to disk
        */
        static bool flushAllChangesInMemoryMapping(FileSystemHelpers::FileMappingResult& viewOfFile);
    };
}
