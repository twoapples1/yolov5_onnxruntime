// 文件以及目录处理

#ifndef __FILE_SYSTEM_H__
#define __FILE_SYSTEM_H__

#include <string>
#include <vector>
 
namespace OrtSamples
{

// 路径是否存在
bool Exists(const std::string &path);

// 路径是否为目录
bool IsDirectory(const std::string &path);

// 是否是路径分隔符(Linux:‘/’,Windows:’\\’)
bool IsPathSeparator(char c);

// 路径拼接
std::string JoinPath(const std::string &base, const std::string &path);

// 创建多级目录,注意：创建多级目录的时候，目标目录是不能有文件存在的
bool CreateDirectories(const std::string &directoryPath);

/** 生成符合指定模式的文件名列表(支持递归遍历)
* 
* pattern: 模式,比如"*.jpg","*.png","*.jpg,*.png"
* addPath：是否包含父路径
* 注意：
    1. 多个模式使用","分割,比如"*.jpg,*.png"
    2. 支持通配符'*','?' ,比如第一个字符是7的所有文件名:"7*.*", 以512结尾的所有jpg文件名："*512.jpg"
    3. 使用"*.jpg"，而不是".jpg"
    4. 空string表示返回所有结果
    5. 不能返回子目录名
*
*/
void GetFileNameList(const std::string &directory, const std::string &pattern, std::vector<std::string> &result, bool recursive, bool addPath);

// 与GetFileNameList的区别在于如果有子目录，在addPath为true的时候会返回子目录路径(目录名最后有"/")
void GetFileNameList2(const std::string &directory, const std::string &pattern, std::vector<std::string> &result, bool recursive, bool addPath);

// 删除文件或者目录,支持递归删除
void Remove(const std::string &directory, const std::string &extension="");

/** 获取路径的文件名和扩展名
 * 
 *  示例：path为D:/1/1.txt,则GetFileName()为1.txt,GetFileName_NoExtension()为1,GetExtension()为.txt,GetParentPath()为D:/1/
*/
std::string GetFileName(const std::string &path);
std::string GetFileName_NoExtension(const std::string &path); 
std::string GetExtension(const std::string &path);
std::string GetParentPath(const std::string &path);

// 拷贝文件
bool CopyFile(const std::string srcPath,const std::string dstPath);

/** 拷贝目录
 * 
 * 示例：CopyDirectories("D:/0/1/2/","E:/3/");实现把D:/0/1/2/目录拷贝到E:/3/目录中(即拷贝完成后的目录结构为E:/3/2/)
 * 注意：
    1.第一个参数的最后不能加”/”
    2.不能拷贝隐藏文件
*/
bool CopyDirectories(std::string srcPath,const std::string dstPath);

}

#endif
