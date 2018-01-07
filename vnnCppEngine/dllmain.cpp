/* Replace "dll.h" with the name of your header */
#include "dll.h"
#include <iostream>

extern "C" {
	
	void* __declspec(dllexport) Create(int nInputs, int nHidden, int nOutputs){
        return new DllClass(nInputs, nHidden, nOutputs);
    }
	//void __declspec(dllexport) Hello(void* p){
	//	((DllClass*)p)->HelloWorld();
	//}
	//void __declspec(dllexport) printWeights(void* p){
	//	((DllClass*)p)->printWeights();
	//}
	void __declspec(dllexport) StaticHello(){
		//MessageBox(0, "Hello World from DLL!\n","Hi", MB_ICONINFORMATION);
		std::cout << "HELLO WORLD!" << std::endl;
	}
	
	double* __declspec(dllexport) get_wInputHidden(void* p){
		return ((DllClass*)p)->wInputHidden;
	}
	double* __declspec(dllexport) get_wHiddenOutput(void* p){
		return ((DllClass*)p)->wHiddenOutput;
	}
	double* __declspec(dllexport) get_outputNeurons(void* p){
		return ((DllClass*)p)->outputNeurons;
	}
	
	void __declspec(dllexport) feedForward(void* p, double* pattern){
		((DllClass*)p)->feedForward(pattern);
	}
	
	double* __declspec(dllexport) FeedResult(void* p, double* pattern){
		((DllClass*)p)->feedForward(pattern);
		return ((DllClass*)p)->outputNeurons;
	}
}

