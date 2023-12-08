#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>
using namespace std;
struct dataStruct {
	int countLayers;
	int* size;
};
class NetWork
{
	int* size;
	int countLayers;
	ActivateFunction actFunc;
	Matrix* weights;
	double** bias; 
	double** neuronsValues;
	double** neuronsErrorsValues;
	double* neuronsBiasValues;
public:
	void Init(dataStruct data);
	void set(double* values);
	void confPrint();
	double directDist();
	int searchMaxIndex(double* value);
	void BackPropagation(double expect);
	void printValues(int L);
	void getNewWeights(double lr);
	void saveCurrentWeights();
	void readCurrentWeights();
};
