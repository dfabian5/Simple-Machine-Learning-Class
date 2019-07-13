#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        progress_bar.h
// DESCRIPTION: contains progress bar class and implementation
// AUTHOR:      Dan Fabian
// DATE:        6/8/2019

#include <iostream>

using std::cout; using std::endl;

////////////////////////////////////////////////////////////////////////////////
//
// PROGRESS BAR
class ProgressBar {
public:
	// constructor
	ProgressBar(unsigned long int total, unsigned int length = 60) : 
		total_(total), length_(length), quarter_(length_ / 4), portion_(0), lengthFilled_(0)
	{ cout << "Progress: ["; }

	void step() 
	{
		++portion_;
		if (portion_ % (total_ / length_) == 0)
		{
			if (lengthFilled_ % quarter_ == 0 && lengthFilled_ != 0 && lengthFilled_ != length_)
				cout << "|";
			else
				cout << "=";
			++lengthFilled_;
		}
		if (portion_ == total_)
			cout << "]" << endl;
	}
	void repost()
	{
		cout << "Progress: [";
		for (size_t i = 0; i < lengthFilled_; ++i)
		{
			if (i % quarter_ == 0 && i != 0 && i != length_)
				cout << "|";
			else
				cout << "=";
		}
	}

private:
	const unsigned int length_; // how many equal signs are going to shown
	const unsigned int total_;
	const unsigned int quarter_;
	unsigned int       portion_;
	unsigned int       lengthFilled_;

};

#endif // PROGRESS_BAR_H