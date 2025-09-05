#ifndef __ZIP_READER_H__
#define __ZIP_READER_H__

#include <string>
#include <filesystem>

namespace NEURAL_NETWORK
{
	class ZipReader 
	{
	public:
		ZipReader();
		~ZipReader();

		ZipReader(const ZipReader&) = delete;
		ZipReader& operator=(const ZipReader&) = delete;

		bool Open(const std::filesystem::path& zipPath);
		bool CheckZipError(int err, const std::string& message) const;
		bool ExtractEntry(const std::filesystem::path& targetPath) const;
		int GoToFirstEntry() const;
		int GoToNextEntry() const;
		bool CheckEndOfFile(int err) const;
		bool CheckOk(int err) const;

	private:
		void* reader_;
	};
}
#endif // __ZIP_READER_H__