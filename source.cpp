#include "NetWork.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <Windows.h>
#include <string>
#include <cstdio> 
using namespace std;
struct basedInfo {
	double* pixels;
	int digit; 
};

dataStruct ReadDataNetWork(string path) {
	dataStruct data{};
	ifstream fin;

	fin.open(path);
	if (!fin.is_open()) {
		cout << "Ошибка чтения!" << path << endl;
	}
	string additionVal;
	int dataCountLayers;
	while (!fin.eof()) {
		fin >> dataCountLayers;
		data.countLayers = dataCountLayers;
		data.size = new int[dataCountLayers];
		for (int i = 0; i < dataCountLayers; i++) {
			fin >> data.size[i];
		}
		fin.close();
		return data;
	}
}

basedInfo* ReadData(string path, const dataStruct& network, int& FinPhoto) {
	basedInfo* data;
	ifstream fin;
	fin.open(path);

	if (!fin.is_open()) {
		cout << "Ошибка чтения файла!" << path << endl;
	}
	else {
		if (path != "normalizedPhoto.txt") {
			cout << "Загружаю... \n"; 
		}
	}
	fin >> FinPhoto;
	cout << "В файле содержатся примеры: " << FinPhoto << endl;
	data = new basedInfo[FinPhoto];
	for (int i = 0; i < FinPhoto; ++i) {
		data[i].pixels = new double[network.size[0]];
	}
	for (int i = 0; i < FinPhoto; ++i) {
		fin >> data[i].digit;
		for (int j = 0; j < network.size[0]; ++j) {
			fin >> data[i].pixels[j];
		}
	}
	fin.close();
	return data;
}
	

void photoProcessing() {
	string filePath;
	cout << "Введи имя файла, который хочешь открыть: ";
	cin >> filePath;
	cv::Mat photo = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
	while (photo.empty()) {
		std::cerr << "Ошибка загрузки файла" << std::endl;
		cout << "Введи имя файла, который хочешь открыть: ";
		string filePath;
		cin >> filePath;
		photo = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
	}
	std::ofstream dataPhoto("normalizedPhoto.txt");
	if (!dataPhoto.is_open()) {
		std::cerr << "Ошибка открытия файла" << std::endl;
	}
	dataPhoto << "1" << std::endl;
	dataPhoto << "1" << std::endl;

	for (int i = 0; i < photo.rows; ++i) {
		for (int j = 0; j < photo.cols; ++j) {
			float brightness = 1 - (static_cast<float>(photo.at<uchar>(i, j)) / 255.0f);
			dataPhoto << brightness << " ";
		}
		dataPhoto << std::endl;
	}

	dataPhoto.close();

}


void editCountExamples() {
	ifstream fin;
	fin.open("trainingSample.txt");

	if (!fin.is_open()) {
		cout << "Ошибка чтения файла для дозаписи!" << endl;
		return;
	}
	
	vector<string> allStrings;

	string pastCountExamples;
	getline(fin, pastCountExamples);
	int intCountExamples = stoi(pastCountExamples) + 1;
	string str;
	while (getline(fin, str)) {
		allStrings.push_back(str);
	}
	
	fin.close();

	remove("trainingSample.txt");
	cout << "Процесс перезаписи файла с учётом нового значения..." << endl;
	ofstream fout;
	fout.open("trainingSample.txt");
	fout << to_string(intCountExamples) + "\n";
	for (int i = 0; i < allStrings.size(); i++) {
		fout << allStrings.at(i) + "\n";
	}	
}

void addNewExample(int trueNumber) {
	ifstream fin;
	fin.open("normalizedPhoto.txt");
	if (!fin.is_open()) {
		cout << "Ошибка чтения файла для дозаписи!" << endl;
		return;
	}
	vector<string> weightsImage;
	string str;
	string trash;
	getline(fin, trash);
	while (getline(fin, str)) {
		cout << str << endl;
		weightsImage.push_back(str);
	}
	fin.close(); 
	fin.open("trainingSample.txt");
	if (!fin.is_open()) {
		cout << "Ошибка дозаписи в файл!" << endl;
		return;
	}
	fin.close();
	ofstream ofs;
	ofs.open("trainingSample.txt", std::ios::app);
	ofs << to_string(trueNumber) + "\n";
	for (int i = 0; i < weightsImage.size(); i++) {
		ofs << weightsImage.at(i) + "\n";
	}
	ofs.close();
	editCountExamples();
}



int main() {
    setlocale(0, "");
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
	NetWork dataNetWork{};
	dataStruct dataNetWorkCONFIG;
	basedInfo* data;

	double predict = 0;
	double rightAnswers = 0;
	int countEpochs = 0;

	string studying;
	chrono::duration<double> allTime;
	dataNetWorkCONFIG = ReadDataNetWork("Settings.txt");
	dataNetWork.Init(dataNetWorkCONFIG);
	dataNetWork.confPrint();
	double rightAnswerExample = 0;

	while (true) {
		cout << "Обучение?" << endl;
		cin >> studying;
		if (studying == "Да") {
			int examples;
			data = ReadData("trainingSample.txt", dataNetWorkCONFIG, examples);
			auto startAllEpochs = chrono::steady_clock::now();
			while (rightAnswers / examples * 100 < 100) {
				rightAnswers = 0;
				auto startCurrentEpoch = chrono::steady_clock::now();
				for (int i = 0; i < examples; ++i) {
					dataNetWork.set(data[i].pixels); 
					rightAnswerExample = data[i].digit;
					predict = dataNetWork.directDist(); 
					if (predict != rightAnswerExample) {
						dataNetWork.BackPropagation(rightAnswerExample);
						dataNetWork.getNewWeights(0.15 * exp(-countEpochs / 20.));
					}
					else {
						rightAnswers++;
					}
				}
				auto stopCurrentEpoch = chrono::steady_clock::now();
				allTime = stopCurrentEpoch - startCurrentEpoch;
				cout << "Количество текущих правильных: " << rightAnswers / examples * 100 << "\t" << "Номер эпохи: " << countEpochs << "\tВремя: " << allTime.count() << endl;
				countEpochs++;
				if (countEpochs == 35) {
					break;
				}
			}
			auto stopAllEpochs = chrono::steady_clock::now();
			allTime = stopAllEpochs - startAllEpochs;
			cout << "Общее количество времени: " << allTime.count() / 60. << " минут" << endl;
			dataNetWork.saveCurrentWeights();
		}
		else {
			dataNetWork.readCurrentWeights();
		}

		cout << "Тестовые?\n";
		string isTest;
		cin >> isTest;

		if (isTest == "Да") {
			int ex_tests;
			basedInfo* data_test;
			data_test = ReadData("testSample.txt", dataNetWorkCONFIG, ex_tests);
			rightAnswers = 0;
			for (int i = 0; i < ex_tests; ++i) {
				dataNetWork.set(data_test[i].pixels);
				predict = dataNetWork.directDist();
				rightAnswerExample = data_test[i].digit;
				if (rightAnswerExample == predict)
					rightAnswers++;
			}
			cout << "Процент правильных ответов: " << rightAnswers / ex_tests * 100 << endl;
		}

		while (true) {
			photoProcessing();
			int testSampleProcessing;
			basedInfo* testSampleData;
			testSampleData = ReadData("normalizedPhoto.txt", dataNetWorkCONFIG, testSampleProcessing);

			for (int i = 0; i < testSampleProcessing; ++i) {
				dataNetWork.set(testSampleData[i].pixels);
				predict = dataNetWork.directDist();
			}

			cout << "Цифра: " << predict << endl;

			cout << "Цифра правильная? ";
			string answer;
			int trueNumber;
			cin >> answer;
			if (answer == "Нет") {
				cout << "Введи правильный ответ: ";
				cin >> trueNumber;
			}
			else {
				trueNumber = predict;
			}
			addNewExample(trueNumber);

		}
	}
	system("pause");
	return 0;
}