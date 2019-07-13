////////////////////////////////////////////////////////////////////////////////
//
// FILE:        main.cpp
// DESCRIPTION: trains a neural network to output its input
// AUTHOR:      Dan Fabian
// DATE:        6/6/2019

#include <iostream>
#include "network.h"
#include <fstream>

using std::cout; using std::endl;
using std::ofstream;

int main()
{
	srand(time(NULL));

	// testing options
	const double STEP_SIZE = .12;
	const int INPUT_OUTPUT_SIZE = 100;
	const int TRAIN_DATA_SIZE = 7000;
	const int TEST_DATA_SIZE = 10000;
	const int TOTAL_DROPS = 3;
	const int TRAINING_PER_DROP = 7000;
	const int DROP_AT_ONCE = 1;
	const double STARTING_LAMBDA = 1;
	const double LAMBDA_DECAY = 2;
	const double TRAINING_GROWTH = 1.55;
	const bool LOAD = false;
	const bool SAVE = !LOAD;
	const bool TRAIN_WITH_TESTING = false;

	// set up network
	Network net(
		vector<size_t>({ INPUT_OUTPUT_SIZE, 8, INPUT_OUTPUT_SIZE }), 
		STEP_SIZE,
		STARTING_LAMBDA
	);

	// setup training data
	cout << "Setting up data..." << endl;
	valarray<ValD> Xtrain(ValD(0.0, INPUT_OUTPUT_SIZE), TRAIN_DATA_SIZE);
	ValD Ytrain(TRAIN_DATA_SIZE);
	ProgressBar tracker(Xtrain.size());
	for (size_t i = 0; i != Xtrain.size(); ++i)
	{
		int ans = rand() % INPUT_OUTPUT_SIZE;
		Xtrain[i][ans] = 1;
		Ytrain[i] = ans;
		tracker.step();
	}

	// setup test data
	valarray<ValD> Xtest(ValD(0.0, INPUT_OUTPUT_SIZE), TEST_DATA_SIZE);
	ValD Ytest(0.0, TEST_DATA_SIZE);
	ProgressBar trackerTwo(Xtest.size());
	for (size_t i = 0; i != Xtest.size(); ++i)
	{
		int ans = rand() % INPUT_OUTPUT_SIZE;
		Xtest[i][ans] = 1;
		Ytest[i] = ans;
		trackerTwo.step();
	}
	cout << "Done setting up data" << endl;

	ofstream out("output.txt");

	// create lambda for training with testing
	auto training = [&](bool test, size_t size) {
		if (test)
			net.train(Xtrain, Ytrain, size,
				out, Xtest, Ytest, 200, 300);
		else
			net.train(Xtrain, Ytrain, size);
	};

	if (!LOAD) // train if not loading from file
	{
		cout << "Begin training..." << endl;
		training(TRAIN_WITH_TESTING, TRAIN_DATA_SIZE);
		
		// dropout
		for (size_t i = 0; i < TOTAL_DROPS; ++i)
		{
			cout << "Dropout " << DROP_AT_ONCE  << " neurons..." << endl;
			net.dropout(1, DROP_AT_ONCE);

			// decay lambda
			net.setLambda(STARTING_LAMBDA / pow(LAMBDA_DECAY, i + 1));

			// train again and grow training
			training(TRAIN_WITH_TESTING, TRAINING_PER_DROP * pow(TRAINING_GROWTH, i));
		}

		cout << "Done training" << endl;
	}
	else
		net.load();

	// test
	cout << "Begin testing..." << endl;
	double correct = net.test(Xtest, Ytest, TEST_DATA_SIZE);
	cout << "Done testing..." << endl;

	// store data
	if (SAVE)
	{
		cout << "Storing data..." << endl;
		net.save();
	}

	cout << "Percent of success: " << correct * 100.0 << "%" << endl;
}

