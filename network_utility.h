#ifndef NETWORK_UTILITY_H
#define NETWORK_UTILITY_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        network_utility.h
// DESCRIPTION: contains helper functions and smaller structs for Network class
// AUTHOR:      Dan Fabian
// DATE:        6/6/2019

#include <valarray>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>
#include <utility>

using std::valarray;
using std::vector;
using std::pair; using std::make_pair;

typedef valarray<double> ValD;

////////////////////////////////////////////////////////////////////////////////
//
// ACTIVATION base
class Activation {
public:
	virtual ValD activate(const ValD &z) = 0;
	virtual ValD prime(const ValD &z) = 0;
};

////////////////////////////////////////////////////////////////////////////////
//
// LINEAR derived
class Linear: public Activation {
public:
	ValD activate(const ValD &z) { return z; }
	ValD prime(const ValD &z) { return ValD(1.0, z.size()); }
};

////////////////////////////////////////////////////////////////////////////////
//
// SIGMOID derived
class Sigmoid: public Activation {
public:
	ValD activate(const ValD &z) { return 1.0 / (1.0 + exp(-z)); }
	ValD prime(const ValD &z) { return activate(z) * (1.0 - activate(z)); }
};

////////////////////////////////////////////////////////////////////////////////
//
// RELU derived
class Relu: public Activation {
public:
	ValD activate(const ValD &z) { return z.apply([](const double &x) { return x <= 0 ? 0.0 : x; }); }
	ValD prime(const ValD &z) { return z.apply([](const double &x) { return x <= 0 ? 0.0 : 1.0; }); }
};

////////////////////////////////////////////////////////////////////////////////
//
// SOFTMAX derived
class Softmax: public Activation {
public:
	ValD activate(const ValD &z) { return z.apply(exp) / z.apply(exp).sum(); }
	ValD prime(const ValD &z) {
		ValD x = z.apply(exp);
		double y = z.apply(exp).sum();
		return (x * y - x * x) / (y * y);
	}
};

////////////////////////////////////////////////////////////////////////////////
//
// LAYER
// notes: weights are indexed as       
//        L1                W[0][0]         L2
//        neuron -------------------------- neuron
//               |          W[1][0]
//               -------------------------- neuron
// and the weights pictured above belong under L2
struct Layer {
	// constructors
	Layer() : size_(0) {}
	Layer(size_t prevLayerNeurons, size_t neurons) :
		weights_(valarray<ValD>(ValD(prevLayerNeurons), neurons)),
		biases_(ValD(neurons)),
		size_(neurons)
	{
		// init seed and create distibution
		std::default_random_engine generator;
		std::normal_distribution<double> distributionOne(0, (1.0 / sqrt(prevLayerNeurons))); // for weights
		std::normal_distribution<double> distributionTwo(0, 1); // for biases

		// init weights with normal distribution with a mean of 0 and SD of 1/sqrt(incoming weights)
		for (size_t i = 0; i != weights_.size(); ++i)
			for (size_t j = 0; j != weights_[i].size(); ++j)
				weights_[i][j] = distributionOne(generator);

		// init biases
		for (size_t i = 0; i != biases_.size(); ++i)
			biases_[i] = distributionTwo(generator);
	}

	// overloaded assignment
	Layer &operator=(const Layer &rhs)
	{
		size_ = rhs.size_;
		weights_ = rhs.weights_;
		biases_ = rhs.biases_;

		return *this;
	}

	valarray<ValD> weights_;
	ValD           biases_;
	size_t         size_;
};

#endif // NETWORK_UTILITY_H
