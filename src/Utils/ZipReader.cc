#include "ZipReader.h"
#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"  
#include "mz_zip_rw.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

NEURAL_NETWORK::ZipReader::ZipReader()
{
	reader_ = mz_zip_reader_create();
}

NEURAL_NETWORK::ZipReader::~ZipReader() 
{
	if (reader_) 
	{
		mz_zip_reader_close(reader_);
		mz_zip_reader_delete(&reader_);
	}
}

bool NEURAL_NETWORK::ZipReader::Open(const std::filesystem::path& zipPath) 
{
	if (!reader_) 
	{
		throw std::logic_error("ZipReader not initialized");
	}
	int err = mz_zip_reader_open_file(reader_, zipPath.string().c_str());
	return err == MZ_OK;
}

bool NEURAL_NETWORK::ZipReader::CheckZipError(int err, 
											  const std::string& message) const
{
	if (err != MZ_OK) 
	{
		std::cerr << "ZIP Error: " << message 
				  << " (code: " << err << ")" 
				  << std::endl;
		return false;
	}
	return true;
}

bool NEURAL_NETWORK::ZipReader::ExtractEntry(const std::filesystem::path& targetPath) const
{
	mz_zip_file* file_info = nullptr;
	int err = mz_zip_reader_entry_get_info(reader_, &file_info);
	if (!CheckZipError(err, "Failed to get entry info")) 
	{
		return false;
	}

	if (!file_info || !file_info->filename) 
	{
		return true;
	}

	std::string filename_str = file_info->filename;
	std::filesystem::path fullPath = targetPath / filename_str;
	std::filesystem::path dir = fullPath.parent_path();

	if (!std::filesystem::exists(dir)) 
	{
		if (!std::filesystem::create_directories(dir)) 
		{
			std::cerr << "Failed to create directory: " 
					  << dir.string() << std::endl;
			return false;
		}
	}

	if (filename_str.back() == '/' || filename_str.back() == '\\') 
	{
		return true;
	}

	err = mz_zip_reader_entry_open(reader_);
	if (!CheckZipError(err, "Failed to open entry: " + filename_str)) 
	{
		return false;
	}

	std::ofstream outFile(fullPath, std::ios::binary);
	if (!outFile.is_open()) 
	{
		std::cerr << "Warning: Failed to create output file: " 
				  << fullPath.string() << std::endl;
		mz_zip_reader_entry_close(reader_);
		return false;
	}

	char buffer[8192];
	int bytesRead = 0;
	bool success = true;
	do {
		bytesRead = mz_zip_reader_entry_read(reader_, buffer, sizeof(buffer));
		if (bytesRead > 0) 
		{
			outFile.write(buffer, bytesRead);
			if (!outFile.good()) 
			{
				outFile.close();
				mz_zip_reader_entry_close(reader_);
				throw std::runtime_error("Failed to write to output file: " + 
					fullPath.string());
			}
		} 
		else if (bytesRead < 0) 
		{
			CheckZipError(bytesRead, 
						  "Failed to read from entry: " + filename_str);
			success = false;
		}
	} while (bytesRead > 0);

	outFile.close();
	mz_zip_reader_entry_close(reader_);
	return success;
}


int NEURAL_NETWORK::ZipReader::GoToFirstEntry() const
{
	if (!reader_) 
	{
		return MZ_PARAM_ERROR;
	}

	return mz_zip_reader_goto_first_entry(reader_);
}

int NEURAL_NETWORK::ZipReader::GoToNextEntry() const
{
	if (!reader_)
	{
		return MZ_PARAM_ERROR;
	}

	return mz_zip_reader_goto_next_entry(reader_);
}

bool NEURAL_NETWORK::ZipReader::CheckEndOfFile(int err) const
{
	return err == MZ_END_OF_STREAM;
}

bool NEURAL_NETWORK::ZipReader::CheckOk(int err) const
{
	return err == MZ_OK;
}