#include "villa_person_tracking/Tracker.h"
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Tracker::Tracker() {
    // setenv("PYTHONPATH","/home/backpack/backpack_ws/src/Robocup_Integrated_Systems/villa_perception/villa_person_tracking/scripts",1);
    Py_Initialize ();
    _import_array();
    pName = PyUnicode_FromString("hands");
    pModule = PyImport_Import(pName);
    if(pModule == nullptr) {
        PyErr_Print();
    }
    pDict = PyModule_GetDict(pModule);
    pFunc = PyDict_GetItemString (pDict, (char*)"initialize"); 
    PyObject_CallObject(pFunc);
 }


void Tracker::getDescriptor(Mat &image) {
    imshow("image",image);
    waitKey(1);
    int row = 0;
    float *p = image.ptr<float>(row);
    npy_intp dims[3] = { image.rows, image.cols, 3 };
    unsigned int nElem = image.rows * image.cols * 3;
    uchar* m = new uchar[nElem];
    std::memcpy(m, image.data, nElem * sizeof(uchar));
    PyObject* mat = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (void*) m);
    args = Py_BuildValue("(O)", mat);

    pFunc = PyDict_GetItemString (pDict, (char*)"process"); 

    PyObject* result = PyObject_CallObject(pFunc, args);
    if(result == nullptr) {
        PyErr_Print();
    }
    float *temp = new float[6];
    memcpy(temp,PyArray_DATA(result),6*sizeof(float));
    descriptor = temp;
}
