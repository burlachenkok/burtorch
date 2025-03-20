#include "burt/fs/include/FileSystemHelpers.h"
#include "burt/fs/include/FileNameHelpers.h"
#include "burt/system/include/Version.h"

#include "gtest/gtest.h"

TEST(burt, FileSystemHelpersGTest)
{
    const std::string cwdOrig = burt::FileSystemHelpers::getCwd();

    EXPECT_TRUE(burt::FileSystemHelpers::chDir(cwdOrig));
    const std::string upFolder = burt::FileNameHelpers::buildFileName(cwdOrig, "/..");
    EXPECT_TRUE(burt::FileSystemHelpers::chDir(upFolder));
    EXPECT_NE(burt::FileSystemHelpers::getCwd(), cwdOrig);
	
    const std::string subFolderName = burt::FileNameHelpers::extractBaseName(cwdOrig);

    EXPECT_TRUE(burt::FileSystemHelpers::isDirExist(subFolderName));
    EXPECT_TRUE(burt::FileSystemHelpers::isDirExist(cwdOrig));

    EXPECT_FALSE(burt::FileSystemHelpers::isFileExist(subFolderName));
    EXPECT_TRUE(burt::FileSystemHelpers::isFileOrFolderExist(subFolderName));
    EXPECT_FALSE(burt::FileSystemHelpers::createDir(subFolderName));

    EXPECT_TRUE(burt::FileSystemHelpers::createDir(subFolderName + "_salt3"));

    EXPECT_TRUE(burt::FileSystemHelpers::removeDir(subFolderName + "_salt3"));
    EXPECT_TRUE(burt::FileSystemHelpers::chDir(cwdOrig));
    EXPECT_EQ(burt::FileSystemHelpers::getCwd(), cwdOrig);
}

TEST(burt, TestFileMapping)
{    
    std::string existFileStr = burt::FileNameHelpers::buildFileName(burt::projectRootDirectory4Build, "../README_1_MIN_TOOLS.md");
    const char* existFile = existFileStr.c_str();   

    {
        burt::FileSystemHelpers::FileMappingResult mapRes = 
            burt::FileSystemHelpers::mapFileToMemory(existFile, true);
        
        EXPECT_TRUE(mapRes.fileSizeInBytes == mapRes.memorySizeInBytes);
        EXPECT_TRUE(burt::FileSystemHelpers::getFileSize(existFile) == mapRes.fileSizeInBytes);

        EXPECT_TRUE(mapRes.isOk);
        EXPECT_TRUE(burt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }

    {
        burt::FileSystemHelpers::FileMappingResult mapRes =
            burt::FileSystemHelpers::mapFileToMemory(existFile, false);

        EXPECT_TRUE(mapRes.fileSizeInBytes == mapRes.memorySizeInBytes);
        EXPECT_TRUE(burt::FileSystemHelpers::getFileSize(existFile) == mapRes.fileSizeInBytes);

        EXPECT_TRUE(mapRes.isOk);
        EXPECT_TRUE(burt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }

    {
        burt::FileSystemHelpers::FileMappingResult mapRes = burt::FileSystemHelpers::mapFileToMemory("not_existing_file.my", true);
        EXPECT_FALSE(mapRes.isOk);
        EXPECT_TRUE(burt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }
}
