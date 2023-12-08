#include "Matrix.h"
void Matrix::Init(int row, int col) {
	this->row = row;
	this->col = col;
	matrix = new double* [row];
	for (int i = 0; i < row; i++)
		matrix[i] = new double[col];

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			matrix[i][j] = 0;
		}
	}
}
void Matrix::setRandValue() {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			matrix[i][j] = ((rand() % 68)) * 0.07 / (row + 10);
		}
	}
}

void Matrix::matrixMultiplication(const Matrix& m1, const double* neuron, int n, double* c, bool isTransported) {
	if (!isTransported) {
		if (m1.col != n)
			throw std::runtime_error("Ошибка в умножении матрицы!\n");
		for (int i = 0; i < m1.row; ++i) {
			double tmp = 0;
			for (int j = 0; j < m1.col; ++j) {
				tmp += m1.matrix[i][j] * neuron[j];
			}
			c[i] = tmp;
		}
	}
	else {
		if (m1.row != n)
			throw std::runtime_error("Ошибка в умножении транспонированной матрицы!\n");
		for (int i = 0; i < m1.col; ++i) {
			double tmp = 0;
			for (int j = 0; j < m1.row; ++j) {
				tmp += m1.matrix[j][i] * neuron[j];
			}
			c[i] = tmp;
		}
	}
}

void Matrix::sumVector(double* a, const double* b, int n) {
	for (int i = 0; i < n; i++)
		a[i] += b[i];
}

double& Matrix::operator()(int i, int j) {
	return matrix[i][j];
}
std::ostream& operator << (std::ostream& os, const Matrix& m) {
	for (int i = 0; i < m.row; ++i) {
		for (int j = 0; j < m.col; j++) {
			os << m.matrix[i][j] << " ";
		}
	}
	return os;
}
std::istream& operator >> (std::istream& is, Matrix& m) {
	for (int i = 0; i < m.row; ++i) {
		for (int j = 0; j < m.col; j++) {
			is >> m.matrix[i][j];
		}
	}
	return is;
}
