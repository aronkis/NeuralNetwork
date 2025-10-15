#include "Helpers.h"
#include "cpr/cpr.h"
#include "ZipReader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

void NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(const std::string& filename,
												  Eigen::MatrixXd& coordinates,
												  Eigen::MatrixXd& classes) 
{
	std::ifstream input_file(filename);

	if (!input_file.is_open()) 
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		coordinates.resize(0, 0); 
		classes.resize(0, 0);
		return;
	}

	std::vector<double> x_coords;
	std::vector<double> y_coords;
	std::vector<int> class_values;

	std::string line;
	int line_num = 0;

	while (std::getline(input_file, line)) 
	{
		line_num++;
		std::stringstream ss(line);
		double x_val, y_val;
		int class_val;

		if (ss >> x_val >> y_val >> class_val) 
		{
			x_coords.push_back(x_val);
			y_coords.push_back(y_val);
			class_values.push_back(class_val);
		} 
		else 
		{
			std::cerr << "Warning: Skipping malformed line " 
					  << line_num << ": '" << line
					  << "' (expected 2 doubles and 1 integer)" << std::endl;
		}
	}

	input_file.close();
	
	long num_rows = x_coords.size();

	if (num_rows > 0)
	{
		coordinates.resize(num_rows, 2);
		classes.resize(num_rows, 1);

		for (long i = 0; i < num_rows; i++)
		{
			coordinates(i, 0) = x_coords[i];
			coordinates(i, 1) = y_coords[i];
			classes(i) = class_values[i];
		}
	}
	else
	{
		std::cout << "No valid data lines found in the file." << std::endl;
		coordinates.resize(0, 0);
		classes.resize(0, 0);
	}
}

void NEURAL_NETWORK::Helpers::Read1DIntoEigen(const std::string& filename,
												  Eigen::MatrixXd& input,
												  Eigen::MatrixXd& output) 
{
	std::ifstream input_file(filename);

	if (!input_file.is_open()) 
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		input.resize(0, 0); 
		output.resize(0, 0);
		return;
	}

	std::vector<double> input_values;
	std::vector<double> output_values;

	std::string line;

	while (std::getline(input_file, line)) 
	{
		std::stringstream ss(line);
		double in_val, out_val;

		if (ss >> in_val >> out_val) 
		{
			input_values.push_back(in_val);
			output_values.push_back(out_val);
		} 
		else 
		{
			std::cerr << "Warning: Skipping malformed line: '" << line
					  << "' (expected 2 doubles)" << std::endl;
		}
	}

	input_file.close();
	
	long num_rows = input_values.size();

	if (num_rows > 0)
	{
		input.resize(num_rows, 1);
		output.resize(num_rows, 1);

		for (long i = 0; i < num_rows; i++)
		{
			input(i, 0) = input_values[i];
			output(i, 0) = output_values[i];
		}
	}
	else
	{
		std::cout << "No valid data lines found in the file." << std::endl;
		input.resize(0, 0);
		output.resize(0, 0);
	}
}

void NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen(const std::string& filename,
                                  				   Eigen::MatrixXd& input,
                                  				   Eigen::MatrixXd& output,
                                  				   char delimiter)
{
    std::ifstream input_file(filename);

    if (!input_file.is_open()) 
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        input.resize(0, 0); 
        output.resize(0, 0);
        return;
    }

    std::vector<double> input1_values;
    std::vector<double> input2_values;
    std::vector<double> output1_values;
    std::vector<double> output2_values;

    std::string line;
    int line_num = 0;

    while (std::getline(input_file, line)) 
    {
        line_num++;
        std::stringstream ss(line);
        std::string token;
        std::vector<double> values;

        while (std::getline(ss, token, delimiter)) 
        {
            try 
            {
                values.push_back(std::stod(token));
            } 
            catch (const std::invalid_argument&) 
            {
                std::cerr << "Warning: Invalid number in line " << line_num 
                          << ": '" << token << "'" << std::endl;
                values.clear();
                break;
            }
        }

        if (values.size() == 4) 
        {
            input1_values.push_back(values[0]);
            input2_values.push_back(values[1]);
            output1_values.push_back(values[2]);
            output2_values.push_back(values[3]);
        } 
        else if (!values.empty()) 
        {
            std::cerr << "Warning: Skipping malformed line " << line_num 
                      << ": '" << line << "' (expected 4 columns)" << std::endl;
        }
    }

    input_file.close();
    
    long num_rows = input1_values.size();

    if (num_rows > 0)
    {
        input.resize(num_rows, 2);
        output.resize(num_rows, 2);

        for (long i = 0; i < num_rows; i++)
        {
            input(i, 0) = input1_values[i];
            input(i, 1) = input2_values[i];
            output(i, 0) = output1_values[i];
            output(i, 1) = output2_values[i];
        }
    }
    else
    {
        std::cout << "No valid data lines found in the file." << std::endl;
        input.resize(0, 0);
        output.resize(0, 0);
    }
}

void NEURAL_NETWORK::Helpers::DownloadData(const std::string url,
					   					   const std::string output_dir,
					   					   const std::string filename)
{
	std::filesystem::path full_path = std::filesystem::path(output_dir) / filename;
	std::filesystem::path dir_path = full_path.parent_path();

	if (!std::filesystem::exists(dir_path))
	{
		std::cout << "Creating directory: " << dir_path.string() << std::endl;
		if (!std::filesystem::create_directories(dir_path))
		{
			std::cerr << "Failed to create directory: " 
					  << dir_path.string() << std::endl;
			return;
		}
		else
		{
			std::cout << "Directory created successfully: " 
					  << dir_path.string() << std::endl;
		}
	}
	else
	{
		std::cout << "Directory already exists: " 
				  << dir_path.string() << std::endl;
	}

	if (std::filesystem::exists(full_path))
	{
		std::cout << "File already exists: " << full_path.string() << std::endl;
		std::cout << "Download cancelled." << std::endl;
		return;
	}

	if (std::filesystem::exists(dir_path / "extracted/"))
	{
		std::cout << "Files already extracted to: " << dir_path / "extracted/" << std::endl 
				  << "Download cancelled." << std::endl;
		return;
	}

	std::ofstream of(full_path, std::ios::binary);
	if (!of) 
	{
		std::cerr << "Cannot open file for writing: " << full_path.string() 
				  << std::endl;
		return;
	}
	
	std::cout << "Downloading " << url 
			  << " to " << full_path.string() << std::endl;
	auto response = cpr::Download(of, cpr::Url{url});
	if (response.status_code != 200)
	{
		std::cerr << "Failed to download file. Status code: " 
				  << response.status_code << std::endl;
		return;
	}

	std::cout << "Download complete." << std::endl;
}

void NEURAL_NETWORK::Helpers::UnzipFile(const std::string& directory,
                                        const std::string& filename,
                                        const std::string& target)
{
    std::filesystem::path zipPath = std::filesystem::path(directory) / filename;
    if (!std::filesystem::exists(zipPath)) 
	{
        std::cerr << "Error: ZIP file not found: " 
				  << zipPath.string() << std::endl;
        return;
    }

    std::filesystem::path targetPath = target;
    if (!std::filesystem::exists(targetPath)) 
	{
        if (!std::filesystem::create_directories(targetPath)) 
		{
            std::cerr << "Failed to create directory: " 
					  << targetPath.string() << std::endl;
            return;
        }
    }

    ZipReader reader;
    if (!reader.Open(zipPath)) 
	{
        return;
    }

    std::cout << "Extracting " << zipPath.string() 
			  << " to " << targetPath.string() << std::endl;

	int err = reader.GoToFirstEntry();

	if (reader.CheckEndOfFile(err))
	{
		reader.CheckZipError(err, "ZIP archive contains no entries.");
		return;
	}

	if (!reader.CheckOk(err)) 
	{
        reader.CheckZipError(err, "Failed to go to first entry");
        return;
    }

    int filesExtracted = 0;
	while (reader.CheckOk(err)) 
	{
		if (reader.ExtractEntry(targetPath)) 
		{
			filesExtracted++;
		}
		else
		{
			std::cerr << "Failed to extract an entry." << std::endl;
			exit(1);
		}
		err = reader.GoToNextEntry();
	}

    if (!reader.CheckEndOfFile(err))
	{
        reader.CheckZipError(err, "Failed to iterate through all entries");
    }

	std::cout << "Successfully extracted " << filesExtracted 
			  << " files" << std::endl;
	if (!std::filesystem::remove(zipPath)) 
	{
			std::cerr << "Warning: Failed to remove zip file: " 
					  << zipPath.string() << std::endl;
	}
	else
	{
		std::cout << "Deleted zipfile" << zipPath.string() << std::endl;
	}
}

void NEURAL_NETWORK::Helpers::FetchData(const std::string url,
					   					const std::string output_dir,
					   					const std::string filename,
					   					const bool unzip)
{
	std::cout << "Downloading data..." << std::endl;
	NEURAL_NETWORK::Helpers::DownloadData(url, output_dir, filename);

	if (unzip) 
	{
		std::filesystem::path extractDir = std::filesystem::path(output_dir) / "extracted";
		if (std::filesystem::exists(extractDir)) 
		{
			std::cout << "Extract directory already exists. Skipping unzip." 
					  << std::endl;
		}
		else
		{
			std::cout << "Unzipping downloaded file..." << std::endl;
			NEURAL_NETWORK::Helpers::UnzipFile(output_dir, 
											   filename, 
											   extractDir.string());
		}
	}

	return;
}

std::vector<std::string> GetFolderContent(const std::string& path, 
										  bool folders = false)
{
	std::vector<std::string> content;
    std::filesystem::path fullPath = std::filesystem::path(path);
    
    if (!std::filesystem::exists(fullPath) || 
		!std::filesystem::is_directory(fullPath)) 
	{
		std::cerr << "Error: Dataset path not found or is not a directory: " 
				  << fullPath.string() 
				  << std::endl;
        return content; 
    }
    if (folders) // can only be folders in the directory
	{
    	for (const auto& entry : std::filesystem::directory_iterator(fullPath)) 
		{
			content.push_back(entry.path().filename().string());
		}
	}
	else // can only be files in the directory
	{
		for (const auto& entry : std::filesystem::directory_iterator(fullPath)) 
		{
			content.push_back(entry.path().filename().string());
		}
	}
	std::sort(content.begin(), content.end());

	return content;
}

Eigen::MatrixXd LoadImage(const std::string& filename, 
						  int& width, 
						  int& height, 
						  int& channels) 
{
    unsigned char* data = stbi_load(filename.c_str(), 
									&width, 
									&height, 
									&channels, 
									0);
    if (!data) 
	{
        width = 0;
        height = 0;
        channels = 0;
        return Eigen::MatrixXd();
    }

	Eigen::MatrixXd image(height, width);
    for (int y = 0; y < height; y++) 
	{
        for (int x = 0; x < width; x++) 
		{
            int index = (y * width + x) * channels;
            image(y, x) = data[index];
        }
    }

	stbi_image_free(data);
    return image;
}

void NEURAL_NETWORK::Helpers::LoadData(const std::string& path,
									   Eigen::Tensor<double, 4>& X_tensor,
									   Eigen::Tensor<double, 2>& y_tensor)
{
	std::vector<std::string> labels = GetFolderContent(path, true);

	int total_images = 0;

	for (const auto& label : labels)
	{
		std::filesystem::path folder = std::filesystem::path(path) / label;
		total_images += GetFolderContent(folder.string()).size();
	}

	if (total_images == 0)
	{
		return;
	}

	int width = 0, height = 0, channels = 0;
	Eigen::MatrixXd first_img;
	std::filesystem::path folder = std::filesystem::path(path) / labels[0];
	std::filesystem::path imgPath = folder / GetFolderContent(folder.string())[0];
	first_img = LoadImage(imgPath.string(), width, height, channels);

	if (first_img.size() == 0)
	{
		std::cerr << "Could not load any images to determine dimensions."
				  << std::endl;
		return;
	}

	X_tensor = Eigen::Tensor<double, 4>(total_images, height, width, channels);
	y_tensor = Eigen::Tensor<double, 2>(total_images, 1);

	int current_image = 0;
	for (const auto& label : labels)
	{
		std::filesystem::path folder = std::filesystem::path(path) / label;
		std::vector<std::string> images = GetFolderContent(folder.string());
		for (const auto& image : images)
		{
			std::filesystem::path imgPath = folder / image;
			Eigen::MatrixXd img = LoadImage(imgPath.string(),
											width,
											height,
											channels);

			if (img.size() == 0)
			{
				std::cerr << "Failed to load image: "
						  << imgPath.string() << std::endl;
				continue;
			}

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					for (int c = 0; c < channels; c++)
					{
						X_tensor(current_image, h, w, c) = img(h, w);
					}
				}
			}

			y_tensor(current_image, 0) = std::stoi(label);
			current_image++;
		}
	}

	if (current_image < total_images) // to account for any skipped images
	{
		Eigen::Tensor<double, 4> resized_X_tensor(current_image, height, width, channels);
		Eigen::Tensor<double, 2> resized_y_tensor(current_image, 1);

		for (int i = 0; i < current_image; i++)
		{
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					for (int c = 0; c < channels; c++)
					{
						resized_X_tensor(i, h, w, c) = X_tensor(i, h, w, c);
					}
				}
			}
			resized_y_tensor(i, 0) = y_tensor(i, 0);
		}

		X_tensor = std::move(resized_X_tensor);
		y_tensor = std::move(resized_y_tensor);
	}

	std::cout << "Loaded " << current_image << " images into tensor format." << std::endl;
}

void NEURAL_NETWORK::Helpers::CreateDataSets(const std::string& dataset_url,
											 const std::string& output_dir,
											 Eigen::Tensor<double, 4>& X_tensor,
											 Eigen::Tensor<double, 2>& y_tensor,
											 Eigen::Tensor<double, 4>& X_test_tensor,
											 Eigen::Tensor<double, 2>& y_test_tensor)
{
	NEURAL_NETWORK::Helpers::FetchData(dataset_url,
									   output_dir,
									   "data.zip",
									   true);

	std::cout << "Loading training data as tensor..." << std::endl;
	NEURAL_NETWORK::Helpers::LoadData(output_dir + "extracted/train",
									  X_tensor,
									  y_tensor);

	std::cout << "Loading test data as tensor..." << std::endl;
	NEURAL_NETWORK::Helpers::LoadData(output_dir + "extracted/test",
									  X_test_tensor,
									  y_test_tensor);

	NEURAL_NETWORK::Helpers::ShuffleData(X_tensor, y_tensor);
	NEURAL_NETWORK::Helpers::ScaleData(X_tensor);
	NEURAL_NETWORK::Helpers::ScaleData(X_test_tensor);
	std::cout << "Data loading and preprocessing complete." << std::endl;
}

void NEURAL_NETWORK::Helpers::ShuffleData(Eigen::Tensor<double, 4>& X_tensor,
                                          Eigen::Tensor<double, 2>& y_tensor)
{
    int batch_size = X_tensor.dimension(0);
    int height = X_tensor.dimension(1);
    int width = X_tensor.dimension(2);
    int channels = X_tensor.dimension(3);

    if (batch_size != y_tensor.dimension(0))
    {
        std::cerr << "Error: X and y batch sizes do not match." << std::endl;
        return;
    }

    std::vector<int> indices(batch_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::Tensor<double, 4> X_shuffled(batch_size, height, width, channels);
    Eigen::Tensor<double, 2> y_shuffled(batch_size, 1);

    for (int i = 0; i < batch_size; i++)
    {
        int shuffled_idx = indices[i];
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    X_shuffled(i, h, w, c) = X_tensor(shuffled_idx, h, w, c);
                }
            }
        }
        y_shuffled(i, 0) = y_tensor(shuffled_idx, 0);
    }

    // Move shuffled data back to original tensors
    X_tensor = std::move(X_shuffled);
    y_tensor = std::move(y_shuffled);
}

void NEURAL_NETWORK::Helpers::ScaleData(Eigen::Tensor<double, 4>& X_tensor)
{
	double min_val = X_tensor(0, 0, 0, 0);
	double max_val = X_tensor(0, 0, 0, 0);

	for (int b = 0; b < X_tensor.dimension(0); b++)
	{
		for (int h = 0; h < X_tensor.dimension(1); h++)
		{
			for (int w = 0; w < X_tensor.dimension(2); w++)
			{
				for (int c = 0; c < X_tensor.dimension(3); c++)
				{
					double val = X_tensor(b, h, w, c);
					min_val = std::min(min_val, val);
					max_val = std::max(max_val, val);
				}
			}
		}
	}

	double range = max_val - min_val;
	if (range > 0)
	{
		for (int b = 0; b < X_tensor.dimension(0); b++)
		{
			for (int h = 0; h < X_tensor.dimension(1); h++)
			{
				for (int w = 0; w < X_tensor.dimension(2); w++)
				{
					for (int c = 0; c < X_tensor.dimension(3); c++)
					{
						X_tensor(b, h, w, c) = 2.0 * (X_tensor(b, h, w, c) - min_val) / range - 1.0;
					}
				}
			}
		}
	}
}

void NEURAL_NETWORK::Helpers::ReadSingleImage(const std::string& filename,
											   Eigen::Tensor<double, 4>& image_tensor)
{
	int width = 0, height = 0, channels = 0;
	Eigen::MatrixXd image = LoadImage(filename, width, height, channels);

	if (image.size() == 0)
	{
		std::cerr << "Failed to load image: " << filename << std::endl;
		image_tensor = Eigen::Tensor<double, 4>(0, 0, 0, 0);
		return;
	}

	image_tensor = Eigen::Tensor<double, 4>(1, height, width, channels);

	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			for (int c = 0; c < channels; c++)
			{
				image_tensor(0, h, w, c) = image(h, w);
			}
		}
	}

	std::cout << "Image loaded successfully: " << filename << std::endl;
	std::cout << "Image dimensions: " << height << "x" << width << "x" << channels << std::endl;
}

Eigen::MatrixXd NEURAL_NETWORK::Helpers::Flatten(const Eigen::MatrixXd& spatial_data)
{
	// This function is for 2D spatial data (e.g., from a flattened tensor)
	// For 4D tensor flattening, we'll need a different approach
	return Eigen::Map<const Eigen::MatrixXd>(spatial_data.data(), spatial_data.rows(), spatial_data.cols());
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Helpers::TensorToTensor2D(const Eigen::Tensor<double, 4>& tensor)
{
	int batch_size = tensor.dimension(0);
	int height = tensor.dimension(1);
	int width = tensor.dimension(2);
	int channels = tensor.dimension(3);

	// Flatten spatial dimensions: (batch, height, width, channels) -> (batch, height*width*channels)
	int flattened_size = height * width * channels;
	Eigen::Tensor<double, 2> tensor2d(batch_size, flattened_size);

	for (int b = 0; b < batch_size; b++)
	{
		int col = 0;
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int c = 0; c < channels; c++)
				{
					tensor2d(b, col++) = tensor(b, h, w, c);
				}
			}
		}
	}

	return tensor2d;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Helpers::MatrixToTensor2D(const Eigen::MatrixXd& matrix)
{
	int rows = matrix.rows();
	int cols = matrix.cols();

	Eigen::Tensor<double, 2> tensor(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			tensor(r, c) = matrix(r, c);
		}
	}

	return tensor;
}

Eigen::Tensor<double, 1> NEURAL_NETWORK::Helpers::RowVectorToTensor1D(const Eigen::RowVectorXd& rowvec)
{
	int size = rowvec.cols();
	Eigen::Tensor<double, 1> tensor(size);

	for (int i = 0; i < size; i++)
	{
		tensor(i) = rowvec(i);
	}

	return tensor;
}

Eigen::MatrixXd NEURAL_NETWORK::Helpers::TensorToMatrix(const Eigen::Tensor<double, 2>& tensor)
{
	int rows = tensor.dimension(0);
	int cols = tensor.dimension(1);

	Eigen::MatrixXd matrix(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			matrix(r, c) = tensor(r, c);
		}
	}

	return matrix;
}

Eigen::RowVectorXd NEURAL_NETWORK::Helpers::TensorToRowVector(const Eigen::Tensor<double, 1>& tensor)
{
	int size = tensor.dimension(0);
	Eigen::RowVectorXd rowvec(size);

	for (int i = 0; i < size; i++)
	{
		rowvec(i) = tensor(i);
	}

	return rowvec;
}