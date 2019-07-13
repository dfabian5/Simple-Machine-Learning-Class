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

using std::valarray;
using std::vector;

typedef valarray<double> ValD;

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
	Layer& operator=(const Layer& rhs)
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

////////////////////////////////////////////////////////////////////////////////
//
// HELPER FUNCTIONS
////////////////////////////////////////
// sigmoid function
ValD sigmoid(const ValD& z)
{
	return 1.0 / (1.0 + exp(-z));
}

////////////////////////////////////////
// sigmoid prime function
ValD sigmoidPrime(const ValD& z)
{
	return sigmoid(z) * (1.0 - sigmoid(z));
}

#endif // NETWORK_UTILITY_H
