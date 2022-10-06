#ifndef TRACKER_H 
#define TRACKER_H
#include <opencv2/core.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>

class Tracker
{
    private:
        PyObject *pName, *pModule, *pDict, *pFunc, *args;
    public:
        Tracker();
        float* get_joints(cv::Mat &image);
};


#endif
