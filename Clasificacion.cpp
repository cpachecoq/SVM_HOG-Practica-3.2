 #include <iostream>
 #include <opencv2/opencv.hpp>
 #include <filesystem>

 namespace fs = std::filesystem;
 using namespace std;
 using namespace cv;
 using namespace cv::ml;

 int main() {
     // Cargar el modelo entrenado
     Ptr<SVM> svm = SVM::load("trained_svm_model.xml");

     // Cargar la imagen del logo a clasificar
     Mat img = imread("pruebas/todas/n8.png", IMREAD_COLOR);
     if (img.empty()) {
         cerr << "No se pudo leer la imagen." << endl;
         return -1;
     }

     // Redimensionar la imagen para que coincida con el tamaño esperado por HOG
     Mat resizedImg;
     resize(img, resizedImg, Size(128, 128));

     // Convertir la imagen a formato de punto flotante y normalizar
     Mat imgDecimal;
     resizedImg.convertTo(imgDecimal, CV_32F, 1.0/255.0);

     // Aplicar el operador Sobel
     Mat gx, gy;
     Sobel(imgDecimal, gx, CV_32F, 1, 0, 1);
     Sobel(imgDecimal, gy, CV_32F, 0, 1, 1);

     // Calcular la magnitud y el ángulo de los gradientes
     Mat magnitud, angulo;
     cartToPolar(gx, gy, magnitud, angulo, true);

     // Calcular las características HOG
     HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
     vector<float> descriptors;
     hog.compute(resizedImg, descriptors);

     // Convertir las características a Mat
     Mat testData = Mat(descriptors).reshape(1, 1);

     // Realizar la predicción
     int response = svm->predict(testData);
     string clase;
     if(response == 1){
         clase = "Youtube";
     } else if(response == 3) {
         clase = "Facebook";
     } else if(response == 4) {
         clase = "Bing";
     } else if(response == 5) {
         clase = "Whatsapp";
    } else if(response == 5) {
         clase = "Netflix";
     } else {
         clase = "Categoria desconocida";
     }
    
     cout << "La clase predicha es: " << clase << endl;

     // Dibujar un rectángulo alrededor de la detección
     Mat deteccion = resizedImg.clone();
     rectangle(deteccion, Point(0, 0), Point(deteccion.cols, deteccion.rows), Scalar(0, 255, 0), 2);

     // Mostrar la imagen con la detección
     namedWindow("Deteccion", WINDOW_AUTOSIZE);
     imshow("Deteccion", deteccion);
     waitKey(0);

     return 0;
 }

