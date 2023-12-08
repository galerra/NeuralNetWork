#pragma once
#include <iostream>
class Matrix
{
	double** matrix;
	int row, col;
public:
	void Init(int row, int col);
	void setRandValue();
	static void matrixMultiplication(const Matrix& m,	const double* b, int n, double* c, bool isTransported);
	static void sumVector(double* a, const double* b, int n);
	double& operator ()(int i, int j);
	friend std::ostream& operator << (std::ostream& os, const Matrix& m);
	friend std::istream& operator >> (std::istream& is, Matrix& m);
};

