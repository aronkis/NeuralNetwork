#pragma once

#include <memory>

namespace NEURAL_NETWORK
{
    class LayerInterface
    {
    public:
        virtual ~LayerInterface() = default;

        // Common interface that all layers must implement
        virtual bool isTrainable() const { return false; }

        // Layer linking (generic)
        virtual void setPrev(std::shared_ptr<LayerInterface> prev) { prev_ = prev; }
        virtual void setNext(std::shared_ptr<LayerInterface> next) { next_ = next; }
        virtual std::shared_ptr<LayerInterface> getPrev() { return prev_.lock(); }
        virtual std::shared_ptr<LayerInterface> getNext() { return next_.lock(); }

    protected:
        std::weak_ptr<LayerInterface> prev_;
        std::weak_ptr<LayerInterface> next_;
    };
}