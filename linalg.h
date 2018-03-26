/*
 *	A collection of linear algebra resources. 
 *	Only suitable for (4x4)-matrices and (4x1)-vector manipulations.
 */

#pragma once

double *normalise(double inputVec[4])
{
	double vectorSum = inputVec[0] + inputVec[1] + inputVec[2] + inputVec[3];

	static double outputVec[4];
	for (int i = 0; i < 4; i++)
	{
		outputVec[i] = inputVec[i] / vectorSum;
	}

	return outputVec;
}

double *softmax(double inputVec[4])
{
	static double outputVec[4];
	for (int i = 0; i < 4; i++)
	{
		outputVec[i] = exp(inputVec[i]);

		double denominator = 0;
		for (int j = 0; j < 4; j++)
		{
			denominator += exp(inputVec[(i + j) % 4]);
		}

		outputVec[i] /= denominator;
	}

	return outputVec;
}

double *multiply(double m1[4][4], double m2[4])
{
	static double m3[4];
	for (int i = 0; i < 4; i++)
	{
		m3[i] = (m1[i][0] * m2[0]) + (m1[i][1] * m2[1]) + (m1[i][2] * m2[2]) + (m1[i][3] * m2[3]);
	}

	return m3;
}