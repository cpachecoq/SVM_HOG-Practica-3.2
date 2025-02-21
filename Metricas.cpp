#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace cv::ml;

vector<float> getHOGFeatures(const Mat& img) {
    Mat resizedImg;
    resize(img, resizedImg, Size(128, 128));
    
    Mat imgDecimal;
    resizedImg.convertTo(imgDecimal, CV_32F, 1.0/255.0);

    Mat gx, gy;
    Sobel(imgDecimal, gx, CV_32F, 1, 0, 1);
    Sobel(imgDecimal, gy, CV_32F, 0, 1, 1);

    Mat magnitud, angulo;
    cartToPolar(gx, gy, magnitud, angulo, true);

    HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    vector<float> descriptors;
    hog.compute(resizedImg, descriptors);
    return descriptors;
}

bool isImageFile(const string& filename) {
    vector<string> validExtensions = {".jpg", ".jpeg", ".png"};
    string ext = fs::path(filename).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end();
}

int main() {
    Ptr<SVM> svm = SVM::load("trained_svm_model.xml");

    vector<string> clases = {"No pertenece", "YouTube", "Netflix", "Bing", "Facebook", "WhatsApp"};
    int num_clases = clases.size();

    map<char, int> initialToClass = {
        {'y', 1},  // YouTube
        {'b', 3},  // Bing
        {'t', 4},  // Netflix
        {'w', 5},  // WhatsApp
        {'n', 0},  // No
        {'f', 2}   // Facebook 
    };

    Mat confusion_matrix = Mat::zeros(num_clases, num_clases, CV_32S);

    string test_dir = "pruebas/todas";

    for (const auto& entry : fs::directory_iterator(test_dir)) {
        string filename = entry.path().filename().string();
        
        if (!isImageFile(filename)) {
            cout << "Saltando archivo no imagen: " << filename << endl;
            continue;
        }

        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        
        if (img.empty()) {
            cout << "No se pudo leer la imagen: " << filename << endl;
            continue;
        }

        char initial = tolower(filename[0]);
        int true_label = -1;

        if (initialToClass.find(initial) != initialToClass.end()) {
            true_label = initialToClass[initial];
        } else {
            cout << "No se pudo determinar la etiqueta verdadera para: " << filename << endl;
            continue;
        }

        vector<float> descriptors = getHOGFeatures(img);
        Mat testData = Mat(descriptors).reshape(1, 1);
        int predicted_label = svm->predict(testData);

        confusion_matrix.at<int>(true_label, predicted_label)++;

        Mat deteccion = img.clone();
        resize(deteccion, deteccion, Size(128, 128));
        rectangle(deteccion, Point(0, 0), Point(deteccion.cols, deteccion.rows), Scalar(0, 255, 0), 2);
        
        string window_name = "Deteccion: " + clases[predicted_label];
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(window_name, deteccion);
        waitKey(100);
    }

    destroyAllWindows();

    cout << "Matriz de Confusion:" << endl;
    cout << "    ";
    for (const auto& clase : clases) cout << clase.substr(0, 3) << " ";
    cout << endl;
    
    for (int i = 0; i < num_clases; i++) {
        cout << clases[i].substr(0, 3) << " ";
        for (int j = 0; j < num_clases; j++) {
            cout << confusion_matrix.at<int>(i, j) << "   ";
        }
        cout << endl;
    }

    for (int i = 0; i < num_clases; i++) {
        int true_positives = confusion_matrix.at<int>(i, i);
        int false_positives = sum(confusion_matrix.col(i))[0] - true_positives;
        int false_negatives = sum(confusion_matrix.row(i))[0] - true_positives;
        
        float precision = true_positives / float(true_positives + false_positives);
        float recall = true_positives / float(true_positives + false_negatives);
        float f1_score = 2 * (precision * recall) / (precision + recall);

        cout << "Metricas para " << clases[i] << ":" << endl;
        cout << "Precision: " << precision << endl;
        cout << "Recall: " << recall << endl;
        cout << "F1-Score: " << f1_score << endl << endl;
    }

    return 0;
}