#ifndef __LAYERBASE_H__
#define __LAYERBASE_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace NEURAL_NETWORK
{
	class LayerBase
	{
	public:
		virtual ~LayerBase() = default;

		LayerBase(const LayerBase&) = delete;
		LayerBase& operator=(const LayerBase&) = delete;

		virtual void forward(const Eigen::Tensor<double, 2>& inputs, bool training) = 0;
		virtual void backward(const Eigen::Tensor<double, 2>& dvalues) = 0;
		virtual Eigen::Tensor<double, 2> predictions() const
		{
			return Eigen::Tensor<double, 2>();
		}

		virtual const Eigen::Tensor<double, 2>& GetOutput() const = 0;
		virtual const Eigen::Tensor<double, 2>& GetDInput() const = 0;

		virtual void SetDInput(const Eigen::Tensor<double, 2>& dinput) = 0;

		virtual std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> GetParameters() const
		{
			return std::make_pair(Eigen::Tensor<double, 2>(), Eigen::Tensor<double, 1>());
		}

		virtual void SetParameters(const Eigen::Tensor<double, 2>& weights, const Eigen::Tensor<double, 1>& biases)
		{}

		virtual double GetWeightRegularizerL1() const { return 0.0; }
		virtual double GetWeightRegularizerL2() const { return 0.0; }
		virtual double GetBiasRegularizerL1() const { return 0.0; }
		virtual double GetBiasRegularizerL2() const { return 0.0; }

		virtual const Eigen::Tensor<double, 2>& GetDWeights() const
		{
			static const Eigen::Tensor<double, 2> empty;
			return empty;
		}

		virtual const Eigen::Tensor<double, 1>& GetDBiases() const
		{
			static const Eigen::Tensor<double, 1> empty;
			return empty;
		}

		virtual const Eigen::Tensor<double, 2>& GetWeightMomentums() const
		{
			static const Eigen::Tensor<double, 2> empty;
			return empty;
		}

		virtual const Eigen::Tensor<double, 1>& GetBiasMomentums() const
		{
			static const Eigen::Tensor<double, 1> empty;
			return empty;
		}

		virtual const Eigen::Tensor<double, 2>& GetWeightCaches() const
		{
			static const Eigen::Tensor<double, 2> empty;
			return empty;
		}

		virtual const Eigen::Tensor<double, 1>& GetBiasCaches() const
		{
			static const Eigen::Tensor<double, 1> empty;
			return empty;
		}

		virtual void SetWeightMomentums(const Eigen::Tensor<double, 2>& weight_momentums)
		{
			(void)weight_momentums; // Suppress unused parameter warning
		}

		virtual void SetBiasMomentums(const Eigen::Tensor<double, 1>& bias_momentums)
		{
			(void)bias_momentums; // Suppress unused parameter warning
		}

		virtual void SetWeightCaches(const Eigen::Tensor<double, 2>& weight_caches)
		{
			(void)weight_caches; // Suppress unused parameter warning
		}

		virtual void SetBiasCaches(const Eigen::Tensor<double, 1>& bias_caches)
		{
			(void)bias_caches; // Suppress unused parameter warning
		}

		virtual void UpdateWeights(Eigen::Tensor<double, 2>& weight_update)
		{
			(void)weight_update; // Suppress unused parameter warning
		}

		virtual void UpdateWeightsCache(Eigen::Tensor<double, 2>& weight_update)
		{
			(void)weight_update; // Suppress unused parameter warning
		}

		virtual void UpdateBiases(Eigen::Tensor<double, 1>& bias_update)
		{
			(void)bias_update; // Suppress unused parameter warning
		}

		virtual void UpdateBiasesCache(Eigen::Tensor<double, 1>& bias_update)
		{
			(void)bias_update; // Suppress unused parameter warning
		}

		virtual const Eigen::Tensor<double, 2>& GetWeights() const
		{
			static const Eigen::Tensor<double, 2> empty;
			return empty;
		}
		virtual const Eigen::Tensor<double, 1>& GetBiases() const
		{
			static const Eigen::Tensor<double, 1> empty;
			return empty;
		}

		void setPrev(const std::shared_ptr<LayerBase>& prev) { prev_ = prev; }
		void setNext(const std::shared_ptr<LayerBase>& next) { next_ = next; }
		std::shared_ptr<LayerBase> getPrev() const { return prev_.lock(); }
		std::shared_ptr<LayerBase> getNext() const { return next_.lock(); }

	protected:
		LayerBase() = default;
		std::weak_ptr<LayerBase> prev_;
		std::weak_ptr<LayerBase> next_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYERBASE_H__
