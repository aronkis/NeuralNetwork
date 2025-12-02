#ifndef __LAYERBASE_H__
#define __LAYERBASE_H__

#include <Eigen/Dense>
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
		virtual Eigen::MatrixXd predictions() const 
		{
			return Eigen::MatrixXd();
		}

		virtual const Eigen::MatrixXd& GetOutput() const = 0;
		virtual const Eigen::MatrixXd& GetDInput() const = 0;

		virtual void SetDInput(const Eigen::MatrixXd& dinput) = 0;

		virtual std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const 
		{
			return std::make_pair(Eigen::MatrixXd(), Eigen::RowVectorXd());
		}
		
		virtual void SetParameters([[maybe_unused]] const Eigen::MatrixXd& weights,
							       [[maybe_unused]] const Eigen::RowVectorXd& biases)
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
		
		virtual void SetWeightMomentums([[maybe_unused]] const Eigen::MatrixXd& weight_momentums) {}
		
		virtual void SetBiasMomentums([[maybe_unused]] const Eigen::RowVectorXd& bias_momentums) {}
		
		virtual void SetWeightCaches([[maybe_unused]] const Eigen::MatrixXd& weight_caches) {}
		
		virtual void SetBiasCaches([[maybe_unused]] const Eigen::RowVectorXd& bias_caches) {}
		
		virtual void UpdateWeights([[maybe_unused]] Eigen::MatrixXd& weight_update) {}
		
		virtual void UpdateWeightsCache([[maybe_unused]] Eigen::MatrixXd& weight_update) {}
		
		virtual void UpdateBiases([[maybe_unused]] Eigen::RowVectorXd& bias_update) {}
		
		virtual void UpdateBiasesCache([[maybe_unused]] Eigen::RowVectorXd& bias_update) {}
		
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

		void SetPrev(const std::shared_ptr<LayerBase>& prev) { prev_ = prev; }
		void SetNext(const std::shared_ptr<LayerBase>& next) { next_ = next; }
		std::shared_ptr<LayerBase> GetPrev() const { return prev_.lock(); }
		std::shared_ptr<LayerBase> GetNext() const { return next_.lock(); }

	protected:
		LayerBase() = default;
		std::weak_ptr<LayerBase> prev_;
		std::weak_ptr<LayerBase> next_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYERBASE_H__
