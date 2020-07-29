#ifndef NETWORK_H
#define NETWORK__H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        network.h
// DESCRIPTION: contains Network class and implementation, uses valarrays
// AUTHOR:      Dan Fabian
// DATE:        6/6/2019

#include "network_utility.h"
#include "progress_bar.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

using std::cout; using std::endl; using std::ostream;
using std::ofstream; using std::ifstream;
using std::string;
using std::istringstream;

typedef std::chrono::high_resolution_clock Clock;

size_t total_epochs = 0;

////////////////////////////////////////////////////////////////////////////////
//
// NETWORK
class Network {
public:
	// constructor
	Network(vector<pair<size_t, Activation*>> layerSizes, double stepConst, double lambda);

	// methods
	void   train(const valarray<ValD>& Xdata, const ValD& Ydata, const size_t epochs,       // trains the whole network
		         ostream& out = cout, const valarray<ValD>& XValdata = valarray<ValD>(),    // params for testing during training
		         const ValD& YValdata = ValD(), size_t testEvery = 0, size_t testingEpochs = 0); 
	double test (const valarray<ValD>& Xdata, const ValD& Ydata, const size_t epochs);      // tests the network and returns a decimal of correct answers / total

	void   dropout   (size_t layer, size_t toDrop);                                         // randomly chooses toDrop amount of neurons to dropout in layer
	void   setLambda (double lambda)                       { lambda_ = lambda; }
	void   setStep   (double step)                         { stepConstant_ = step; }
	void   save      (string name = "save.txt")     const;                                  // stores layer sizes, weights, and biases in a text file
	void   load      (string name = "save.txt");                                            // loads layers, weights, and biases from a text file

private:
	// helper functions
	void backPropagation    (const ValD& alpha, const ValD& Yvalue); // uses backprop to adjust weights and biases
	ValD forwardPropagation (const ValD& inputs);                    // returns a valarray of output layer activations

	vector<Activation *> activations_;
	vector<Layer>        layers_;
	vector<ValD>         z_; // need to store z values after each forward prop to be used in back prop alg
	double               stepConstant_;
	double               lambda_;
	size_t               trainingSetSize_;
};

////////////////////////////////////////////////////////////////////////////////
//
// NETWORK functions
////////////////////////////////////////
// constructor
Network::Network(vector<pair<size_t, Activation *>> layerSizes, double stepConst, double lambda) :
	activations_(vector<Activation *>(layerSizes.size())),
	layers_(vector<Layer>(layerSizes.size())),
	z_(vector<ValD>(layerSizes.size())),
	stepConstant_(stepConst),
	lambda_(lambda),
	trainingSetSize_(0)
{
	// init activations vector, first activation is null since it is never used
	for (size_t i = 1; i != layers_.size(); ++i)
		activations_[i] = layerSizes[i].second;

	// init layers, start from 1 since 0 is the input layer which doesn't have weights or biases
	layers_[0].size_ = layerSizes[0].first;
	for (size_t i = 1; i != layers_.size(); ++i)
		layers_[i] = Layer(layerSizes[i - 1].first, layerSizes[i].first);
}

////////////////////////////////////////
// forward propagation, returns a valarray of output layer activations
ValD Network::forwardPropagation(const ValD & inputs)
{
	// alpha is the activation from the previous neuron
	// input layer
	ValD alpha = inputs;

	// also store inputs in z_[0] for use in backprop function
	z_[0] = inputs;

	// begin progatating forward, layer 0 is the input layer so start at layer 1
	for (size_t l = 1; l != layers_.size(); ++l)
	{
		ValD z(layers_[l].size_);
		for (size_t j = 0; j != layers_[l].size_; ++j) // finding z for the j-th neuron in the l-th layer
			z[j] = (layers_[l].weights_[j] * alpha).sum() + layers_[l].biases_[j];

		// save z values
		z_[l] = z;

		// get activations
		alpha = activations_[l]->activate(z);
	}

	return alpha;
}

////////////////////////////////////////
// back propagation algorithm to adjust weights and biases in each layer
void Network::backPropagation(const ValD & alpha, const ValD & Yvalue)
{
	// init vector of deltas for each layer
	const size_t L = layers_.size() - 1; // final layer
	valarray<ValD> delta(L + 1);

	// begin with delta in the output layer
	delta[L] = Yvalue * (1.0 - alpha) - alpha * (1.0 - Yvalue);

	// now propagate backward to find deltas
	for (size_t l = L - 1; l > 0; --l)
	{
		ValD deltaSum(0.0, layers_[l].size_);
		for (size_t k = 0; k != layers_[l + 1].size_; ++k)
			deltaSum += layers_[l + 1].weights_[k] * delta[l + 1][k] * activations_[l]->prime(z_[l]);

		delta[l] = deltaSum;
	}

	// adjust weights and biases
	for (size_t l = 1; l != layers_.size(); ++l)
	{
		layers_[l].biases_ += stepConstant_ * delta[l];
		ValD activation = sigmoid(z_[l - 1]);

		for (size_t j = 0; j != layers_[l].size_; ++j)
		{
			double deltaAndRatio = stepConstant_ * delta[l][j];
			double regularization = lambda_ / trainingSetSize_;

			if (l - 1 != 0)
				layers_[l].weights_[j] += deltaAndRatio * activation + regularization * layers_[l].weights_[j];
			else
				layers_[l].weights_[j] += deltaAndRatio * z_[l - 1] + regularization * layers_[l].weights_[j]; // doesn't need sigmoid since its the raw input
		}
	}
}

////////////////////////////////////////
// trains the whole network
void Network::train(const valarray<ValD> & Xdata, const ValD & Ydata, const size_t epochs,
	                ostream& out, const valarray<ValD>& XValdata, const ValD& YValdata, size_t testEvery, size_t testingEpochs)
{
	trainingSetSize_ = Xdata.size();
	auto t1 = Clock::now();
	ProgressBar tracker(epochs);

	for (size_t ep = 0; ep < epochs; ++ep) // iterations
	{
		size_t index = rand() % Xdata.size(); // select a random piece of data to train with
		auto alpha = forwardPropagation(Xdata[index]);
		ValD ans(0.0, alpha.size());
		ans[Ydata[index]] = 1.0;

		if (testEvery != 0)
			if (ep % testEvery == 0) // need seperate statement so there isn't a remainder by 0
			{
				cout << endl << "Intermediate accuracy testing..." << endl; 
				out << total_epochs << endl
					<< test(XValdata, YValdata, testingEpochs) << endl;
				cout << "Continue training... " << endl;
				tracker.repost();
			}

		backPropagation(alpha, ans);
		tracker.step();
		++total_epochs;
	}

	auto t2 = Clock::now();
	cout << "Training took "
		<< std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
		<< " seconds" << endl;
}

////////////////////////////////////////
// tests the network and returns a decimal of correct answers / total
double Network::test(const valarray<ValD> & Xdata, const ValD & Ydata, const size_t epochs)
{
	auto t1 = Clock::now();
	ProgressBar tracker(epochs);

	size_t success = 0;
	for (size_t i = 0; i != epochs; ++i)
	{
		auto alpha = forwardPropagation(Xdata[i]);

		// find argmax
		size_t argmax = 0;
		for (size_t j = 1; j != alpha.size(); ++j)
			if (alpha[argmax] < alpha[j])
				argmax = j;

		if (argmax == Ydata[i])
			++success;

		tracker.step();
	}

	auto t2 = Clock::now();
	cout << "Testing took "
		<< std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
		<< " seconds" << endl;
	return double(success) / double(epochs);
}

////////////////////////////////////////
// randomly chooses toDrop amount of neurons to dropout in layer
void Network::dropout(size_t layer, size_t toDrop)
{
	if (layer == 0 || layer >= layers_.size() - 1) // can't dropout in the input layer or output layer
		return;

	// select neurons to keep
	vector<int> neuronsToDrop(toDrop, -1); // list of neuron indicies to drop
	size_t i = 0;
	while (i < neuronsToDrop.size())
	{
		int possibleIndex = rand() % layers_[layer].size_; // select random index
		if (std::find(neuronsToDrop.begin(), neuronsToDrop.end(), possibleIndex) == neuronsToDrop.end()) // make sure the index isn't repeated
		{
			neuronsToDrop[i] = possibleIndex;
			++i;
		}
	}

	// changing layer
	valarray<ValD> newWeights(ValD(layers_[layer - 1].size_), layers_[layer].size_ - toDrop);
	ValD newBiases(layers_[layer].size_ - toDrop);
	for (size_t j = 0, k = 0; j < layers_[layer].size_; ++j)
		if (std::find(neuronsToDrop.begin(), neuronsToDrop.end(), j) == neuronsToDrop.end()) // if index j isnt to be dropped
		{
			newWeights[k] = layers_[layer].weights_[j];
			newBiases[k] = layers_[layer].biases_[j];
			++k;
		}

	layers_[layer].weights_ = newWeights;
	layers_[layer].biases_ = newBiases;
	layers_[layer].size_ = layers_[layer].size_ - toDrop;

	// changing next layer
	newWeights = valarray<ValD>(ValD(layers_[layer].size_), layers_[layer + 1].size_);
	for (size_t j = 0; j < newWeights.size(); ++j)
		for (size_t k = 0, p = 0; k < newWeights[j].size() + toDrop; ++k)
			if (std::find(neuronsToDrop.begin(), neuronsToDrop.end(), k) == neuronsToDrop.end()) // if index k isnt to be dropped
			{
				newWeights[j][p] = layers_[layer + 1].weights_[j][k];
				++p;
			}

	layers_[layer + 1].weights_ = newWeights;
}

////////////////////////////////////////
// stores layer sizes, weights, and biases in a text file
void Network::save(string name) const
{
	ofstream store(name);

	// output layer sizes first seperated by a space
	for (size_t i = 0; i < layers_.size(); ++i)
		store << layers_[i].size_ << ' ';
	store << endl;

	// now store weights
	for (size_t i = 1; i < layers_.size(); ++i)
		for (size_t j = 0; j < layers_[i].size_; ++j)
		{
			for (size_t k = 0; k < layers_[i].weights_[j].size(); ++k)
				store << layers_[i].weights_[j][k] << ' ';
			store << endl;
		}

	// store biases
	for (size_t i = 1; i < layers_.size(); ++i)
	{
		for (size_t j = 0; j < layers_[i].size_; ++j)
			store << layers_[i].biases_[j] << ' ';
		store << endl;
	}
}

////////////////////////////////////////
// loads layer sizes, weights, and biases from a text file
void Network::load(string name)
{
	ifstream input(name);

	if (!input.is_open())
	{
		cout << "Couldn't load network." << endl;
		return;
	}
	cout << "Loading network..." << endl;

	// get layer sizes
	string line; // store the whole line
	std::getline(input, line);
	istringstream iss(line);
	string word; // store each word
	vector<size_t> layerSizes;
	while (iss >> word)
		layerSizes.push_back(stoi(word));

	// now set layers
	layers_ = vector<Layer>(layerSizes.size());
	layers_[0].size_ = layerSizes[0];
	for (size_t i = 1; i != layers_.size(); ++i)
		layers_[i] = Layer(layerSizes[i - 1], layerSizes[i]);

	// get weights
	string weight;
	for (size_t i = 0; i < layers_.size(); ++i)
		for (size_t j = 0; j < layers_[i].size_; ++j)
			for (size_t k = 0; k < layers_[i].weights_[j].size(); ++k)
			{
				input >> weight;
				layers_[i].weights_[j][k] = stod(weight);
			}

	// get biases
	string bias;
	for (size_t i = 0; i < layers_.size(); ++i)
		for (size_t j = 0; j < layers_[i].size_; ++j)
		{
			input >> bias;
			layers_[i].biases_[j] = stod(bias);
		}

	cout << "Network loaded" << endl;
}

#endif // NETWORK_H
