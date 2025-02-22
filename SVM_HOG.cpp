#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace ml;



int main() {
    try {
        string basePath = "./positivas";
        vector<int> labels;

        HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

        int label = 0;
        int totalImages = 0;

        // Archivo temporal para almacenar las características HOG
        ofstream hogFile("hog_features.tmp", ios::binary);
        if (!hogFile) {
            cerr << "No se pudo abrir el archivo temporal para escribir" << endl;
            return -1;
        }

        for (const auto& category : fs::directory_iterator(basePath)) {
            if (fs::is_directory(category)) {
                cout << "Procesando categoría: " << category.path().filename().string() << endl;
                int imagesInCategory = 0;
                for (const auto& file : fs::directory_iterator(category.path())) {
                    Mat img = imread(file.path().string(), IMREAD_COLOR);
                    if (!img.empty()) {
                        Mat resizedImg;
                        resize(img, resizedImg, Size(128, 128));

                        vector<float> descriptors;
                        hog.compute(resizedImg, descriptors);

                        // Escribir los descriptores en el archivo
                        hogFile.write(reinterpret_cast<char*>(descriptors.data()), descriptors.size() * sizeof(float));

                        labels.push_back(label);
                        imagesInCategory++;
                        totalImages++;
                    } else {
                        cout << "No se pudo leer la imagen: " << file.path().filename() << endl;
                    }
                }
                cout << "Categoría " << label << ": " << imagesInCategory << " imágenes procesadas." << endl;
                if (imagesInCategory > 0) {
                    label++;
                } else {
                    cout << "Advertencia: No se procesaron imágenes en la categoría " << category.path().filename() << endl;
                }
            }
        }

        hogFile.close();

        cout << "Total de imágenes procesadas: " << totalImages << endl;

        if (totalImages == 0) {
            cerr << "No hay datos para entrenar." << endl;
            return -1;
        }

        // Leer las características HOG del archivo temporal
        ifstream hogFileRead("hog_features.tmp", ios::binary);
        if (!hogFileRead) {
            cerr << "No se pudo abrir el archivo temporal para leer" << endl;
            return -1;
        }

        Mat trainData(totalImages, 34020, CV_32F);  
        for (int i = 0; i < totalImages; ++i) {
            hogFileRead.read(reinterpret_cast<char*>(trainData.ptr(i)), 34020 * sizeof(float));
            if (i % 5 == 0) {
                //cout << "Leída fila " << i << " de " << totalImages << endl;
            }
        }

        hogFileRead.close();

        cout << "Dimensiones de trainData: " << trainData.rows << " x " << trainData.cols << endl;

        if (trainData.rows != labels.size()) {
            cerr << "El número de características (" << trainData.rows << ") no coincide con el número de etiquetas (" << labels.size() << ")." << endl;
            return -1;
        }

        Mat labelsMat(labels);
        
        if (labelsMat.empty()) {
            cerr << "labelsMat está vacía." << endl;
            return -1;
        }

        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

        cout << "Iniciando entrenamiento..." << endl;
        
        try {
            svm->train(trainData, ROW_SAMPLE, labelsMat);
            cout << "Entrenamiento completado." << endl;
        } catch (const cv::Exception& e) {
            cerr << "Error de OpenCV durante el entrenamiento: " << e.what() << endl;
            return -1;
        }

        svm->save("trained_svm_model.xml");

        cout << "Modelo guardado. Entrenamiento completado con " << trainData.rows << " muestras." << endl;

        // Eliminar el archivo temporal
        fs::remove("hog_features.tmp");

    } catch (const exception& e) {
        cerr << "Se produjo una excepción: " << e.what() << endl;
        return -1;
    } catch (...) {
        cerr << "Se produjo una excepción desconocida." << endl;
        return -1;
    }

    return 0;
}


