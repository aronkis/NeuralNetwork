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
        virtual Eigen::MatrixXd predictions() const = 0;

        virtual const Eigen::MatrixXd& GetOutput() const = 0;
        virtual const Eigen::MatrixXd& GetDInput() const = 0;
        
		virtual void SetDInput(const Eigen::MatrixXd& dinput) = 0;

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
