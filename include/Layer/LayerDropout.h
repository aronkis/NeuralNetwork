#ifndef __LAYER_DROPOUT_H__
#define __LAYER_DROPOUT_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class LayerDropout : public LayerBase
	{
	public:
		LayerDropout(double rate);
		~LayerDropout() = default;

		void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 2>& dvalues) override;
		Eigen::Tensor<double, 2> predictions() const override;

		const Eigen::Tensor<double, 2>& GetOutput() const override;
		const Eigen::Tensor<double, 2>& GetDInput() const override;
		double GetRate() const;

		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;

	private:
		Eigen::Tensor<double, 2> inputs_;

		Eigen::Tensor<double, 2> output_;

		Eigen::Tensor<double, 2> d_inputs_;

		Eigen::Tensor<double, 2> mask_;

		double rate_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYER_DROPOUT_H__