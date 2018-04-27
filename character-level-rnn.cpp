/*
 * Vanilla recurrent neural network that performs basic character generation.
 * A C++ implementation of the example given in Andrej Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks".
 * Reference: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

#include <iostream>
#include <string>
#include "linalg.h"

double *vectify(char inputChar); //used to convert a string into a vector (with one hot encoding)

class RNN
{
private:
	double Wxh[4][4]; //input -> hidden weight matrix with parameters (hidden size, vocab size)
	double Whh[4][4]; //hidden -> hidden weight matrix with parameters (hidden size, hidden size)
	double Why[4][4]; //hidden -> output weight matrix with parameters (vocab size, hidden size)
	double h[4]; //hidden vector
	double hPrev[4]; //hidden vector from the previous step
public:
	//randomly initialise the weights and zero the hidden vector
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

		zeroh();
	}

	void zeroh()
	{
		for (int i = 0; i < 4; i++)
			h[i] = 0;
	}

	double *step(double *x)
	{
		//compute the matrix multiplications using the hidden state h and the input vector x
		double *multiPtr;

		double entryOne[4];
		multiPtr = multiply(Whh, h);
		for (int i = 0; i < 4; i++)
			entryOne[i] = multiPtr[i];

		double entryTwo[4];
		multiPtr = multiply(Wxh, x);
		for (int i = 0; i < 4; i++)
			entryTwo[i] = multiPtr[i];

		//update the hidden state
		for (int i = 0; i < 4; i++)
		{
			hPrev[i] = h[i];
			h[i] = tanh(entryOne[i] + entryTwo[i]);
		}

		double *y;
		y = multiply(Why, h);

		return y;
	}

	//cross entropy loss between the target and the output
	double error(double *targetVec, double *outputVec)
	{
		//normalise outputVec to get the probablility of each character
		double *p;
		p = softmax(outputVec);

		//define our error by the cross entropy loss
		double error = 0;
		for (int i = 0; i < 4; i++)
			error -= (targetVec[i] * log(p[i])) + ((1 - targetVec[i]) * log(1 - p[i]));

		return error;
	}

	//adjust the weights via backpropagation
	void backProp(double *inputVec, double *outputVec, double *targetVec)
	{
		//calculate the error differentials E_Why, E_Whh and E_Wxh
		double *p;
		p = softmax(outputVec);

		double E_y[4]; //this differential formula follows from the chain rule: E_y = E_p * p_y
		for (int i = 0; i < 4; i++)
		{
			if (targetVec[i] == 1)
				E_y[i] = p[i] - 1;
			else
				E_y[i] = p[i];
		}

		//calculate E_Why
		double y_Why[4];
		for (int i = 0; i < 4; i++)
			y_Why[i] = tanh(h[i]);

		double E_Why[4][4] //by the chain rule, E_Why = E_y * y_Why (N.B., this is applied elementwise)
				= { { E_y[0] * y_Why[0], E_y[0] * y_Why[1], E_y[0] * y_Why[2], E_y[0] * y_Why[3] },
				    { E_y[1] * y_Why[0], E_y[1] * y_Why[1], E_y[1] * y_Why[2], E_y[1] * y_Why[3] },
				    { E_y[2] * y_Why[0], E_y[2] * y_Why[1], E_y[2] * y_Why[2], E_y[2] * y_Why[3] },
				    { E_y[3] * y_Why[0], E_y[3] * y_Why[1], E_y[3] * y_Why[2], E_y[3] * y_Why[3] } };

		//tool to calculate the error differentials of Whh and Wxh
		double y_h[4];
		for (int i = 0; i < 4; i++)
			y_h[i] = 1 - (tanh(h[i]) * tanh(h[i]));

		double *multiPtr;
		multiPtr = multiply(Why, y_h);

		//calculate E_Whh
		double y_Whh[4];
		for (int i = 0; i < 4; i++)
			y_Whh[i] = multiPtr[i] * hPrev[i];

		double E_Whh[4][4] //E_Whh = E_y * y_Whh (N.B., this is applied elementwise)
				= { { E_y[0] * y_Whh[0], E_y[0] * y_Whh[1], E_y[0] * y_Whh[2], E_y[0] * y_Whh[3] },
				    { E_y[1] * y_Whh[0], E_y[1] * y_Whh[1], E_y[1] * y_Whh[2], E_y[1] * y_Whh[3] },
				    { E_y[2] * y_Whh[0], E_y[2] * y_Whh[1], E_y[2] * y_Whh[2], E_y[2] * y_Whh[3] },
				    { E_y[3] * y_Whh[0], E_y[3] * y_Whh[1], E_y[3] * y_Whh[2], E_y[3] * y_Whh[3] } };

		//calculate E_Wxh
		double y_Wxh[4];
		for (int i = 0; i < 4; i++)
			y_Wxh[i] = multiPtr[i] * inputVec[i];

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
};

int main()
{
	std::string trainingSet = "hello";

	RNN layer1;
	RNN layer2;
	layer1.initialise();
	layer2.initialise();

	std::cout << "Enter the depth of training:" << std::endl;
	int depth;
	std::cin >> depth;
	while (depth > 0)
	{
		for (unsigned int i = 0; i < trainingSet.size() - 1; i++)
		{
			double *vectorPtr;

			double inputVec[4];
			vectorPtr = vectify(trainingSet[i]);
			for (int i = 0; i < 4; i++)
				inputVec[i] = vectorPtr[i]; //though cumbersome, these definitions avoid pointer-related errors

			double targetVec[4];
			vectorPtr = vectify(trainingSet[i + 1]);
			for (int i = 0; i < 4; i++)
				targetVec[i] = vectorPtr[i];

			double outputVec1[4];
			vectorPtr = layer1.step(inputVec);
			for (int i = 0; i < 4; i++)
				outputVec1[i] = vectorPtr[i];

			double outputVec2[4];
			vectorPtr = layer2.step(softmax(outputVec1));
			for (int i = 0; i < 4; i++)
				outputVec2[i] = vectorPtr[i];

			//N.B., backpropogation between neuron layers can take various routes
			layer1.backProp(inputVec, outputVec1, targetVec);
			layer2.backProp(outputVec1, outputVec2, targetVec);
		}

		//reset the hidden vector to its original state
		layer1.zeroh();
		layer2.zeroh();

		depth--;
	}

	std::cout << "Training complete." << std::endl;
	std::cout << "Enter 'h', 'e', 'l', 'o' or type 'quit' to quit:" << std::endl;
	std::string inputString;
	std::cin >> inputString;
	while (inputString != "quit")
	{
		double *vectorPtr;

		double inputVec[4];
		vectorPtr = vectify(inputString[0]);
		for (int i = 0; i < 4; i++)
			inputVec[i] = vectorPtr[i];

		double outputVec1[4];
		vectorPtr = layer1.step(inputVec);
		for (int i = 0; i < 4; i++)
			outputVec1[i] = vectorPtr[i];

		double outputVec2[4];
		vectorPtr = layer2.step(softmax(outputVec1));
		for (int i = 0; i < 4; i++)
			outputVec2[i] = vectorPtr[i];

		//register the most probable output
		int maxIndex;
		double maxValue = 0;
		for (int i = 0; i < 4; i++)
		{
			if (outputVec2[i] > maxValue)
			{
				maxIndex = i;
				maxValue = outputVec2[i];
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
		std::cin >> inputString;
	}

	return 0;
}

double *vectify(char inputChar)
{
	double outputVec[4] = { 0, 0, 0, 0 };

	switch (inputChar)
	{
		case 'h': outputVec[0] = 1;
			break;
		case 'e': outputVec[1] = 1;
			break;
		case 'l': outputVec[2] = 1;
			break;
		case 'o': outputVec[3] = 1;
			break;
	}

	return outputVec;
}
