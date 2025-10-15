#include "LossBinaryCrossEntropy.h"

void NEURAL_NETWORK::LossBinaryCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions,
													 const Eigen::Tensor<double, 2>& targets)
{
    int rows = predictions.dimension(0);
    int cols = predictions.dimension(1);

    // Initialize output tensor if needed
    if (output_.size() == 0 || output_.dimension(0) != rows || output_.dimension(1) != 1)
    {
        output_ = Eigen::Tensor<double, 2>(rows, 1);
    }

    // Clip predictions and compute binary cross entropy manually
    for (int r = 0; r < rows; r++)
    {
        double row_loss = 0.0;
        for (int c = 0; c < cols; c++)
        {
            // Clip predictions to avoid log(0)
            double pred_clipped = std::max(1e-7, std::min(1.0 - 1e-7, predictions(r, c)));

            // Binary cross entropy loss formula
            double loss = -(targets(r, c) * std::log(pred_clipped) +
                           (1.0 - targets(r, c)) * std::log(1.0 - pred_clipped));
            row_loss += loss;
        }
        output_(r, 0) = row_loss / cols; // Mean across columns
    }
}

void NEURAL_NETWORK::LossBinaryCrossEntropy::backward(const Eigen::Tensor<double, 2>& d_values,
													  const Eigen::Tensor<double, 2>& targets)
{
	int samples = d_values.dimension(0);
	int outputs = d_values.dimension(1);

    // Initialize d_inputs_ tensor if needed
    if (d_inputs_.size() == 0 || d_inputs_.dimension(0) != samples || d_inputs_.dimension(1) != outputs)
    {
        d_inputs_ = Eigen::Tensor<double, 2>(samples, outputs);
    }

    // Compute gradients manually for tensors
    for (int s = 0; s < samples; s++)
    {
        for (int o = 0; o < outputs; o++)
        {
            // Clip d_values to avoid division by zero
            double d_val_clipped = std::max(1e-7, std::min(1.0 - 1e-7, d_values(s, o)));

            // Binary cross entropy gradient formula
            double gradient = -(targets(s, o) / d_val_clipped -
                               (1.0 - targets(s, o)) / (1.0 - d_val_clipped)) / outputs;

            d_inputs_(s, o) = gradient / samples;
        }
    }
}