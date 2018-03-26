/*
 * Vanilla recurrent neural network that performs basic character generation.
 * A C++ implementation of the example given in Andrej Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks".
 * Reference: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
*/

#include <iostream>
#include <string>
#include "linalg.h"

double Wxh[4][4]; //input -> hidden weight matrix with parameters (hidden size, vocab size)
double Whh[4][4]; //hidden -> hidden weight matrix with parameters (hidden size, hidden size)
double Why[4][4]; //hidden -> output weight matrix with parameters (vocab size, hidden size)
double h[4]; //hidden vector

double *vectify(char inputChar); //used to convert a string into a vector (with one hot encoding)
double *step(double *x); //RNN step function
double error(double *inputVec, double *outputVec); //cross entropy loss between the target and the output

void initialise(); //randomly initialise the weights and zero the hidden vector
void backProp(double *inputVec, double *outputVec, double *targetVec, double *oldh); //adjust the weights via backpropagation

int main()
{
	std::string inputString = "hello";
	initialise();

	std::cout << "Enter the depth of training:" << std::endl;
	int depth;
	std::cin >> depth;

	while (depth > 0)
	{
		for (int i = 0; i < inputString.size() - 1; i++)
		{
			double *vectorPtr;

			double inputVec[4];
			vectorPtr = vectify(inputString[i]);
			for (int i = 0; i < 4; i++)
			{
				inputVec[i] = vectorPtr[i]; //though cumbersome, these definitions avoid pointer-related errors
			}

			double targetVec[4];
			vectorPtr = vectify(inputString[i + 1]);
			for (int i = 0; i < 4; i++)
			{
				targetVec[i] = vectorPtr[i];
			}

			double oldh[4];
			for (int i = 0; i < 4; i++)
			{
				oldh[i] = h[i];
			}

			double *outputVec;
			outputVec = step(inputVec);

			backProp(inputVec, outputVec, targetVec, oldh);
		}

		//reset the hidden vector to its original state
		for (int i = 0; i < 4; i++)
		{
			h[i] = 0;
		}

		depth--;
	}

	std::cout << "Training complete.\nEnter 'h', 'e', 'l' or 'o':" << std::endl;

	std::string testData;
	std::cin >> testData;
	while (testData != "quit")
	{
		double *vectorPtr;

		double inputVec[4];
		vectorPtr = vectify(testData[0]);
		for (int i = 0; i < 4; i++)
		{
			inputVec[i] = vectorPtr[i];
		}

		double *outputVec;
		outputVec = step(inputVec);

		int maxIndex;
		double maxValue = 0;
		for (int i = 0; i < 4; i++)
		{
			if (outputVec[i] > maxValue)
			{
				maxIndex = i;
				maxValue = outputVec[i];
			}
		}

		std::cout << "Output: " << std::endl;
		switch (maxIndex)
		{
			case 0: std::cout << "h";
				break;
			case 1: std::cout << "e";
				break;
			case 2: std::cout << "l";
				break;
			case 3: std::cout << "o";
				break;
		}
		std::cout << std::endl;

		std::cin >> testData;
	}

	return 0;
}

void initialise()
{
	//initialise the matrices with random values
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
		{
			//get random numbers from the interval [0, 200]
			double Rxh = (rand() % 201);
			double Rhh = (rand() % 201);
			double Rhy = (rand() % 201);

			//map: [0, 200] -> [-1, 1], with 2 decimal place precision
			Wxh[i][j] = (Rxh / 100) - 1;
			Whh[i][j] = (Rhh / 100) - 1;
			Why[i][j] = (Rhy / 100) - 1;
		}

	//zero the hidden vector
	for (int i = 0; i < 4; i++)
	{
		h[i] = 0;
	}
}

double *vectify(char inputChar)
{
	double outputVec[4] = { 0, 0, 0, 0 };
	
	if (inputChar == 'h')
	{
		outputVec[0] = 1;
	}
	else if (inputChar == 'e')
	{
		outputVec[1] = 1;
	}
	else if (inputChar == 'l')
	{
		outputVec[2] = 1;
	}
	else if (inputChar == 'o')
	{
		outputVec[3] = 1;
	}

	return outputVec;
}

double *step(double *x)
{
	//compute the matrix multiplications using the old state h and the input vector x
	double *multiPtr;

	double entryOne[4];
	multiPtr = multiply(Whh, h);
	for (int i = 0; i < 4; i++)
	{
		entryOne[i] = multiPtr[i];
	}

	double entryTwo[4];
	multiPtr = multiply(Wxh, x);
	for (int i = 0; i < 4; i++)
	{
		entryTwo[i] = multiPtr[i];
	}

	//update the hidden state
	for (int i = 0; i < 4; i++)
	{
		h[i] = tanh(entryOne[i] + entryTwo[i]);
	}

	double *y;
	y = multiply(Why, h);

	return y;
}

double error(double *inputVec, double *outputVec)
{
	double targetVec[4] = { 0, 0, 0, 0 };

	if (inputVec[0] == 1)
	{
		targetVec[1] = 1;
	}
	else if (inputVec[1] == 1)
	{
		targetVec[2] = 1;
	}
	else if (inputVec[2] == 1)
	{
		targetVec[3] = 1;
	}
	else if (inputVec[3] == 1)
	{
		targetVec[0] = 1;
	}

	//normalise outputVec to get the probablility of each character
	double *p;
	p = softmax(outputVec);

	//define our error by the cross entropy loss
	double error = 0;
	for (int i = 0; i < 4; i++)
	{
		error -= (targetVec[i] * log(p[i])) + ((1 - targetVec[i]) * log(1 - p[i]));
	}

	return error;
}

void backProp(double *inputVec, double *outputVec, double *targetVec, double *oldh)
{
	//calculate the error differentials E_Why, E_Whh and E_Wxh
	double *p;
	p = softmax(outputVec);

	double E_y[4]; //this differential formula follows from the chain rule: E_y = E_p * p_y
	for (int i = 0; i < 4; i++)
	{
		if (targetVec[i] == 1)
		{
			E_y[i] = p[i] - 1;
		}
		else
		{
			E_y[i] = p[i];
		}
	}

	//calculate E_Why
	double y_Why[4];
	for (int i = 0; i < 4; i++)
	{
		y_Why[i] = tanh(h[i]);
	}

	double E_Why[4][4] //by the chain rule, E_Why = E_y * y_Why (N.B., this is applied elementwise)
				= { { E_y[0] * y_Why[0], E_y[0] * y_Why[1], E_y[0] * y_Why[2], E_y[0] * y_Why[3] },
					{ E_y[1] * y_Why[0], E_y[1] * y_Why[1], E_y[1] * y_Why[2], E_y[1] * y_Why[3] },
					{ E_y[2] * y_Why[0], E_y[2] * y_Why[1], E_y[2] * y_Why[2], E_y[2] * y_Why[3] },
					{ E_y[3] * y_Why[0], E_y[3] * y_Why[1], E_y[3] * y_Why[2], E_y[3] * y_Why[3] } };

	//tool to calculate the error differentials of Whh and Wxh
	double y_h[4];
	for (int i = 0; i < 4; i++)
	{
		y_h[i] = 1 - (tanh(h[i]) * tanh(h[i]));
	}

	double *multiPtr;
	multiPtr = multiply(Why, y_h);

	//calculate E_Whh
	double y_Whh[4];
	for (int i = 0; i < 4; i++)
	{
		y_Whh[i] = multiPtr[i] * oldh[i];
	}

	double E_Whh[4][4] //E_Whh = E_y * y_Whh (N.B., this is applied elementwise)
				= { { E_y[0] * y_Whh[0], E_y[0] * y_Whh[1], E_y[0] * y_Whh[2], E_y[0] * y_Whh[3] },
					{ E_y[1] * y_Whh[0], E_y[1] * y_Whh[1], E_y[1] * y_Whh[2], E_y[1] * y_Whh[3] },
					{ E_y[2] * y_Whh[0], E_y[2] * y_Whh[1], E_y[2] * y_Whh[2], E_y[2] * y_Whh[3] },
					{ E_y[3] * y_Whh[0], E_y[3] * y_Whh[1], E_y[3] * y_Whh[2], E_y[3] * y_Whh[3] } };

	//calculate E_Wxh
	double y_Wxh[4];
	for (int i = 0; i < 4; i++)
	{
		y_Wxh[i] = multiPtr[i] * inputVec[i];
	}

	double E_Wxh[4][4] //E_Wxh = E_y * y_Wxh (N.B., this is applied elementwise)
				= { { E_y[0] * y_Wxh[0], E_y[0] * y_Wxh[1], E_y[0] * y_Wxh[2], E_y[0] * y_Wxh[3] },
					{ E_y[1] * y_Wxh[0], E_y[1] * y_Wxh[1], E_y[1] * y_Wxh[2], E_y[1] * y_Wxh[3] },
					{ E_y[2] * y_Wxh[0], E_y[2] * y_Wxh[1], E_y[2] * y_Wxh[2], E_y[2] * y_Wxh[3] },
					{ E_y[3] * y_Wxh[0], E_y[3] * y_Wxh[1], E_y[3] * y_Wxh[2], E_y[3] * y_Wxh[3] } };

	//nudge each weight in the appropriate direction with scaling factor learnRate
	double learnRate = 0.5;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
		{
			Wxh[i][j] -= learnRate * E_Wxh[i][j];
			Whh[i][j] -= learnRate * E_Whh[i][j];
			Why[i][j] -= learnRate * E_Why[i][j];
		}
}
