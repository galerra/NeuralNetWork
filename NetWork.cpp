#include "NetWork.h"
void NetWork::Init(dataStruct data) {
	srand(time(NULL));

	countLayers = data.countLayers;
	size = new int[countLayers];
	for (int i = 0; i < countLayers; i++)
		size[i] = data.size[i];

	weights = new Matrix[countLayers - 1];
	bias = new double* [countLayers - 1];
	for (int i = 0; i < countLayers - 1; i++) {
		weights[i].Init(size[i + 1], size[i]);
		bias[i] = new double[size[i + 1]];
		weights[i].setRandValue();
		for (int j = 0; j < size[i + 1]; j++) {
			bias[i][j] = ((rand() % 58)) * 0.05 / (size[i] + 10);
		}
	}
	neuronsValues = new double* [countLayers]; 
	neuronsErrorsValues = new double* [countLayers];
	for (int i = 0; i < countLayers; i++) {
		neuronsValues[i] = new double[size[i]];
		neuronsErrorsValues[i] = new double[size[i]];
	}
	neuronsBiasValues = new double[countLayers - 1];
	for (int i = 0; i < countLayers - 1; i++)
		neuronsBiasValues[i] = 1;
}
void NetWork::confPrint() {
	cout << "---------------------------------------------------------------\n";
	cout << "У нейросети " << countLayers << " слоя \nРазмер массива: ";
	for (int i = 0; i < countLayers; i++) {
		cout << size[i] << " ";
	}
	cout << "\n--------------------------------------------------------------\n\n";
}
void NetWork::set(double* values) {
	for (int i = 0; i < size[0]; i++) {
		neuronsValues[0][i] = values[i];
	}
}

int NetWork::searchMaxIndex(double* value) {
	double max = value[0];
	int maxIndex = 0;
	double additionVar;
	for (int i = 1; i < size[countLayers - 1]; i++) {
		additionVar = value[i];
		if (additionVar > max) {
			maxIndex = i;
			max = additionVar;
		}
	}
	return maxIndex;
}
double NetWork::directDist() {
	for (int i = 1; i < countLayers; ++i) {
		Matrix::matrixMultiplication(weights[i - 1], neuronsValues[i - 1], size[i - 1], neuronsValues[i], false);
		Matrix::sumVector(neuronsValues[i], bias[i - 1], size[i]);
		actFunc.useFunction(neuronsValues[i], size[i]);
	}
	int pred = searchMaxIndex(neuronsValues[countLayers - 1]);
	return pred;
}

void NetWork::printValues(int L) {
	for (int j = 0; j < size[L]; j++) {
		cout << j << " " << neuronsValues[L][j] << endl;
	}
}

void NetWork::BackPropagation(double expect) {
	for (int i = 0; i < size[countLayers - 1]; i++) {
		if (i != int(expect))
			neuronsErrorsValues[countLayers - 1][i] = -neuronsValues[countLayers - 1][i] * actFunc.useFunctionDerivative(neuronsValues[countLayers - 1][i]);
		else
			neuronsErrorsValues[countLayers - 1][i] = (1.0 - neuronsValues[countLayers - 1][i]) * actFunc.useFunctionDerivative(neuronsValues[countLayers - 1][i]);
	}
	for (int i = countLayers - 2; i > 0; i--) {
		Matrix::matrixMultiplication(weights[i], neuronsErrorsValues[i + 1], size[i + 1], neuronsErrorsValues[i], true);
		for (int j = 0; j < size[i]; j++)
			neuronsErrorsValues[i][j] *= actFunc.useFunctionDerivative(neuronsValues[i][j]);
	}
}
void NetWork::getNewWeights(double ratio) {
	for (int i = 0; i < countLayers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			for (int k = 0; k < size[i]; ++k) {
				weights[i](j, k) += neuronsValues[i][k] * neuronsErrorsValues[i + 1][j] * ratio;
			}
		}
	}
	for (int i = 0; i < countLayers - 1; i++) {
		for (int k = 0; k < size[i + 1]; k++) {
			bias[i][k] += neuronsErrorsValues[i + 1][k] * ratio;
		}
	}
}
void NetWork::saveCurrentWeights() {
	ofstream fout;
	fout.open("Weights.txt");
	if (!fout.is_open()) {
		cout << "Ошибка чтения файла весов";
		system("pause");
	}
	for (int i = 0; i < countLayers - 1; ++i)
		fout << weights[i] << " ";

	for (int i = 0; i < countLayers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fout << bias[i][j] << " ";
		}
	}
	cout << "Веса сохранены \n";
	fout.close();
}
void NetWork::readCurrentWeights() {
	ifstream fin;
	fin.open("Weights.txt");
	if (!fin.is_open()) {
		cout << "Ошибка чтения файла весов";
		system("pause");
	}
	for (int i = 0; i < countLayers - 1; ++i) {
		fin >> weights[i];
	}
	for (int i = 0; i < countLayers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fin >> bias[i][j];
		}
	}
	fin.close();
}