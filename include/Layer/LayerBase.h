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
		
		virtual void forward(const Eigen::MatrixXd& inputs, bool training) = 0;
		virtual void backward(const Eigen::MatrixXd& dvalues) = 0;

		// Tensor-based interface for CNN layers
		virtual bool SupportsTensorInterface() const { return false; }
		virtual void forward(const Eigen::Tensor<double, 4>& inputs, bool training)
		{
			// Default implementation converts tensor to matrix and calls matrix version
			// This allows non-CNN layers to work without modification
			(void)inputs; (void)training; // Suppress unused parameter warnings
		}
		virtual void backward(const Eigen::Tensor<double, 4>& dvalues)
		{
			// Default implementation converts tensor to matrix and calls matrix version
			(void)dvalues; // Suppress unused parameter warning
		}
		virtual Eigen::MatrixXd predictions() const 
		{
			return Eigen::MatrixXd();
		}

		virtual const Eigen::MatrixXd& GetOutput() const = 0;
		virtual const Eigen::MatrixXd& GetDInput() const = 0;

		virtual void SetDInput(const Eigen::MatrixXd& dinput) = 0;

		// Tensor-based getters for CNN layers
		virtual const Eigen::Tensor<double, 4>& GetTensorOutput() const
		{
			static const Eigen::Tensor<double, 4> empty;
			return empty;
		}
		virtual const Eigen::Tensor<double, 4>& GetTensorDInput() const
		{
			static const Eigen::Tensor<double, 4> empty;
			return empty;
		}
		virtual void SetTensorDInput(const Eigen::Tensor<double, 4>& dinput)
		{
			(void)dinput; // Suppress unused parameter warning
		}

		virtual std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const 
		{
			return std::make_pair(Eigen::MatrixXd(), Eigen::RowVectorXd());
		}
		
		virtual void SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases)
		{}

		virtual double GetWeightRegularizerL1() const { return 0.0; }
		virtual double GetWeightRegularizerL2() const { return 0.0; }
		virtual double GetBiasRegularizerL1() const { return 0.0; }
		virtual double GetBiasRegularizerL2() const { return 0.0; }

		virtual const Eigen::MatrixXd& GetDWeights() const 
		{
			static const Eigen::MatrixXd empty;
			return empty;
		}
		
		virtual const Eigen::RowVectorXd& GetDBiases() const 
		{
			static const Eigen::RowVectorXd empty;
			return empty;
		}

		virtual const Eigen::MatrixXd& GetWeightMomentums() const 
		{
			static const Eigen::MatrixXd empty;
			return empty;
		}
		
		virtual const Eigen::RowVectorXd& GetBiasMomentums() const 
		{
			static const Eigen::RowVectorXd empty;
			return empty;
		}
		
		virtual const Eigen::MatrixXd& GetWeightCaches() const 
		{
			static const Eigen::MatrixXd empty;
			return empty;
		}
		
		virtual const Eigen::RowVectorXd& GetBiasCaches() const 
		{
			static const Eigen::RowVectorXd empty;
			return empty;
		}
		
		virtual void SetWeightMomentums(const Eigen::MatrixXd& weight_momentums) 
		{
			(void)weight_momentums; // Suppress unused parameter warning
		}
		
		virtual void SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums) 
		{
			(void)bias_momentums; // Suppress unused parameter warning
		}
		
		virtual void SetWeightCaches(const Eigen::MatrixXd& weight_caches) 
		{
			(void)weight_caches; // Suppress unused parameter warning
		}
		
		virtual void SetBiasCaches(const Eigen::RowVectorXd& bias_caches) 
		{
			(void)bias_caches; // Suppress unused parameter warning
		}
		
		virtual void UpdateWeights(Eigen::MatrixXd& weight_update) 
		{
			(void)weight_update; // Suppress unused parameter warning
		}
		
		virtual void UpdateWeightsCache(Eigen::MatrixXd& weight_update) 
		{
			(void)weight_update; // Suppress unused parameter warning
		}
		
		virtual void UpdateBiases(Eigen::RowVectorXd& bias_update) 
		{
			(void)bias_update; // Suppress unused parameter warning
		}
		
		virtual void UpdateBiasesCache(Eigen::RowVectorXd& bias_update) 
		{
			(void)bias_update; // Suppress unused parameter warning
		}
		
		virtual const Eigen::MatrixXd& GetWeights() const
		{
			static const Eigen::MatrixXd empty;
			return empty;
		}
		virtual const Eigen::RowVectorXd& GetBiases() const
		{
			static const Eigen::RowVectorXd empty;
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
