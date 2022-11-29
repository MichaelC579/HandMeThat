#include "tracker.h"
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
    PyObject_CallObject(pFunc, NULL);
 }


float* Tracker::get_joints(Mat &image) {
    if(!image.empty()) {
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
        PyErr_Print();
        if((int) PyList_Size(result) != 0) {
            float *temp = new float[9];
            char* a11;
            PyObject *ptemp, *objectsRepresentation;
            for(int i = 0; i < 9; i++) {
                ptemp = PyList_GetItem(result,i);
                temp[i] = PyFloat_AsDouble(ptemp);
            }
            return temp;
        }
    }
    return NULL;
}
