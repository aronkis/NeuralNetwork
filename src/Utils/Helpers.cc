#include "Helpers.h"
#include "cpr/cpr.h"
#include "ZipReader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <random>
#include <algorithm>

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

void NEURAL_NETWORK::Helpers::ReadCSVMatrix(const std::string& filename,
											 Eigen::MatrixXd& matrix,
											 char delimiter)
{
    std::ifstream input_file(filename);

    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        matrix.resize(0, 0);
        return;
    }

    std::string line;
    std::vector<std::vector<double>> data;
    int line_num = 0;

    while (std::getline(input_file, line))
    {
        line_num++;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, delimiter))
        {
            try
            {
                row.push_back(std::stod(cell));
            }
            catch (const std::invalid_argument&)
            {
                std::cerr << "Warning: Invalid number in line " << line_num
                          << ": '" << cell << "'" << std::endl;
                continue;
            }
        }

        if (!row.empty())
        {
            data.push_back(row);
        }
    }

    input_file.close();

    if (data.empty())
    {
        std::cout << "No valid data found in " << filename << std::endl;
        matrix.resize(0, 0);
        return;
    }

    long num_rows = data.size();
    long num_cols = data[0].size();

    matrix.resize(num_rows, num_cols);
    for (long i = 0; i < num_rows; i++)
    {
        for (long j = 0; j < num_cols; j++)
        {
            matrix(i, j) = data[i][j];
        }
    }

    std::cout << "Loaded " << filename << ": " << matrix.rows()
              << " samples, " << matrix.cols() << " features" << std::endl;
}

void NEURAL_NETWORK::Helpers::ReadCSVLabels(const std::string& filename,
											 Eigen::VectorXi& labels)
{
    std::ifstream input_file(filename);

    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        labels.resize(0);
        return;
    }

    std::string line;
    std::vector<int> label_data;
    int line_num = 0;

    while (std::getline(input_file, line))
    {
        line_num++;
        try
        {
            label_data.push_back(std::stoi(line));
        }
        catch (const std::invalid_argument&)
        {
            std::cerr << "Warning: Invalid label in line " << line_num
                      << ": '" << line << "'" << std::endl;
        }
    }

    input_file.close();

    if (label_data.empty())
    {
        std::cout << "No valid labels found in " << filename << std::endl;
        labels.resize(0);
        return;
    }

    long num_labels = label_data.size();
    labels.resize(num_labels);

    for (long i = 0; i < num_labels; i++)
    {
        labels(i) = label_data[i];
    }

    std::cout << "Loaded " << filename << ": " << labels.size()
              << " labels" << std::endl;
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
									   Eigen::MatrixXd& X,
									   Eigen::MatrixXd& y) 
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
        X.resize(0, 0);
        y.resize(0, 0);
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
        X.resize(0, 0);
        y.resize(0, 0);
        return;
    }

	int feature_size = width * height;
    X.resize(total_images, feature_size);
    y.resize(total_images, 1);

	int current_row = 0;
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

			Eigen::RowVectorXd correct_flattened(img.size());
			int idx = 0;
			for (int y = 0; y < img.rows(); y++)
			{
				for (int x = 0; x < img.cols(); x++)
				{
					correct_flattened(idx++) = img(y, x);
				}
			}
			X.row(current_row) = correct_flattened;
			y(current_row, 0) = std::stoi(label);
			current_row++;
		}
	}

	if (current_row < total_images) 
	{
        X.conservativeResize(current_row, feature_size);
        y.conservativeResize(current_row, 1);
    }

    std::cout << "Loaded " << X.rows() << " images." << std::endl;
}

void NEURAL_NETWORK::Helpers::CreateDataSets(const std::string& dataset_url,
											 const std::string& output_dir,
											 Eigen::MatrixXd& X,
											 Eigen::MatrixXd& y,
											 Eigen::MatrixXd& X_test,
											 Eigen::MatrixXd& y_test)
{
	NEURAL_NETWORK::Helpers::FetchData(dataset_url,
                                       output_dir,
                                       "data.zip",
                                       true);
	std::vector<Eigen::MatrixXd> X_vect;
	std::vector<int> y_vect;
	std::vector<Eigen::MatrixXd> X_test_vect;
	std::vector<int> y_test_vect;

	std::cout << "Loading training data..." << std::endl;
    NEURAL_NETWORK::Helpers::LoadData(output_dir + "extracted/train", 
									  X, 
									  y);
									  
    std::cout << "Loading test data..." << std::endl;
    NEURAL_NETWORK::Helpers::LoadData(output_dir + "extracted/test", 
									  X_test, 
									  y_test);

	NEURAL_NETWORK::Helpers::ShuffleData(X, y);
	NEURAL_NETWORK::Helpers::ScaleData(X);
    NEURAL_NETWORK::Helpers::ScaleData(X_test);
}

void NEURAL_NETWORK::Helpers::ShuffleData(Eigen::MatrixXd& X,
                                          Eigen::MatrixXd& y) 
{
    if (X.rows() != y.rows()) 
    {
        std::cerr << "Error: X and y row counts do not match." << std::endl;
        return;
    }

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.rows());
    perm.setIdentity();
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.indices().data(), 
				 perm.indices().data() + perm.indices().size(), 
				 g);

    X = perm * X;
    y = perm * y;
}

void NEURAL_NETWORK::Helpers::ScaleData(Eigen::MatrixXd& X) 
{
	double mean = X.mean();
	double std_dev = std::sqrt((X.array() - mean).square().mean());
	if (std_dev > 0) 
	{
		X = (X.array() - mean) / std_dev;
	}
}

void NEURAL_NETWORK::Helpers::ReadSingleImage(const std::string& filename, 
											   Eigen::MatrixXd& image)
{    
	int width = 0, height = 0, channels = 0;
	image = LoadImage(filename, width, height, channels);

	if (image.size() == 0) 
	{
		std::cerr << "Failed to load image: " << filename << std::endl;
		return;
	}
	image = Eigen::Map<Eigen::RowVectorXd>(image.data(), image.size());
	std::cout << "Image loaded successfully: " << filename << std::endl;
}