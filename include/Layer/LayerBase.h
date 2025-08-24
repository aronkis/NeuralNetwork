#ifndef __LAYERBASE_H__
#define __LAYERBASE_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK 
{
    
    class LayerBase 
    {
    public:
        virtual ~LayerBase() = default;
        
        virtual void forward(const Eigen::MatrixXd& inputs, bool training) = 0;
        virtual void backward(const Eigen::MatrixXd& dvalues) = 0;
        virtual const Eigen::MatrixXd& GetOutput() const = 0;
        virtual const Eigen::MatrixXd& GetDInput() const = 0;
        virtual void SetDInput(const Eigen::MatrixXd& dinput) = 0;
        virtual Eigen::MatrixXd predictions() const = 0;

        LayerBase(const LayerBase&) = delete;
        LayerBase& operator=(const LayerBase&) = delete;

        void setPrev(void* prev) { prev_ = prev; }
        void setNext(void* next) { next_ = next; }
        void* getPrev() const { return prev_; }
        void* getNext() const { return next_; }
        
    protected:
        LayerBase() = default;
        void* prev_ = nullptr;
        void* next_ = nullptr;
    };

} // namespace NEURAL_NETWORK

#endif // __LAYERBASE_H__
