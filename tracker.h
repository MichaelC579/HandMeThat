#ifndef TRACKER_H 
#define TRACKER_H
#include <opencv2/core.hpp>
#include <Python.h>
#include <numpy/arrayobjects.h>

class Tracker
{
    private:
        PyObject *pName, *pModule, *pDict, *pFunc, *args;
    public:
        Tracker();
        void getDescriptor(cv::Mat &image);
};


#endif
