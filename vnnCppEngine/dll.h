#include <iostream>
#include <math.h>
#include <random>

static double* createEmptyArr(int out, int in){
	double* re = new double[in * out];
	//double** re = new double*[in];
	//for(int i = 0; i < in; i++){
	//	re[i] = new double[out];
	//}
	return re;
}
static double getElementAt(int x, int y, int width, double* weights){
	return weights[x * width + y];
}
static void setElem(int x, int y, int width, double* weights, double value){
	weights[x * width + y] = value;
}

class DllClass
{
	public:
		DllClass(int _nInputs, int _nHidden, int _nOutputs) : DllClass(_nInputs, _nHidden, _nOutputs, 
			createEmptyArr(_nInputs + 1, _nHidden), 
			createEmptyArr(_nHidden + 1, _nOutputs)) 
		{
			
		}
		
		DllClass(int _nInputs, int _nHidden, int _nOutputs, double* _wInputHidden, double* _wHiddenOutput){
			this->nInput = _nInputs;
			this->nHidden = _nHidden;
			this->nOutput = _nOutputs;
			
			inputNeurons = new double[nInput + 1];
			inputNeurons[nInput] = 1;

			hiddenNeurons = new double[nHidden + 1];
			hiddenNeurons[nHidden] = 1;

			outputNeurons = new double[nOutput];
			
			wInputHidden = _wInputHidden;
			wHiddenOutput = _wHiddenOutput;
			
			this->RandomizeUniform();
		}
		
		int nInput; 
		int nHidden;
		int nOutput;
		double* inputNeurons;
		double* hiddenNeurons;
		double* outputNeurons;
		double* wInputHidden;
		double* wHiddenOutput;
		void setWIH(int x, int y, double val){
			wInputHidden[x * (nInput + 1) + y] = val;
		}
		void setWHO(int x, int y, double val){
			wHiddenOutput[x * (nHidden + 1) + y] = val;
		}
		
		void feedForward(double* pattern){
			//set input neurons to input values
            for(int i = 0; i < nInput; i++) { inputNeurons[i] = pattern[i]; }
			
			/*
			for(int rec = 0; rec < nHidden; rec++)
			{
                //get weighted sum of pattern and bias neuron
                double sum = 0.0;
				for(int give = 0; give <= nInput; give++) { sum += inputNeurons[give] * wInputHidden[give][rec]; }
                hiddenNeurons[rec] = sum;

				//set to result of sigmoid
				hiddenNeurons[rec] = 1 / (1 + exp(-hiddenNeurons[rec])); //activation function
			}

			//Calculating Output Layer values - include bias neuron
			//--------------------------------------------------------------------------------------------------------
			for(int rec = 0; rec < nOutput; rec++)
			{
				//get weighted sum of pattern and bias neuron
                double sum = 0.0;
				for(int give = 0; give <= nHidden; give++) { sum += hiddenNeurons[give] * wHiddenOutput[give][rec]; }
                outputNeurons[rec] = sum;

				//set to result of sigmoid
				outputNeurons[rec] = 1 / (1 + exp(-outputNeurons[rec])); //activation function
			}
			*/
			
			mult(wInputHidden, inputNeurons, hiddenNeurons, nInput + 1, nHidden);
			mult(wHiddenOutput, hiddenNeurons, outputNeurons, nHidden + 1, nOutput);
		}

		inline void mult(double * A, double *x, double *y, int insize, int outsize)
		{
			double ytemp;
			double *Apos = &A[0];
	
			for(int i=0; i < outsize; i++)
			{
				double *xpos = &x[0];
				ytemp=0;
	
				for(int j=0; j < insize; j++)
				{
					ytemp += (*Apos++) * (*xpos++);
				}
		
				y[i] =  1.0 / (1.0 + exp(-ytemp));
			}	
		}
		
		void RandomizeUniform(double mult1 = 5, double mult2 = 3.5)
        {
        	std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(-1.0,1.0);
            //set weights between input and hidden 		
            //--------------------------------------------------------------------------------------------------------
            double limit = sqrt(3.0 / nInput) * mult1;
            for (int i = 0; i <= nInput; i++)
            {
                for (int j = 0; j < nHidden; j++)
                {
                    //set weights to random values
                    setWIH(j, i, (distribution(generator)) * limit);
					//setElem(j, i, nInput + 1, wInputHidden, (distribution(generator)) * limit);
                }
            }

            //set weights between input and hidden
            //--------------------------------------------------------------------------------------------------------
            limit = sqrt(3.0 / nHidden) * mult2;
            for (int i = 0; i <= nHidden; i++)
            {
                for (int j = 0; j < nOutput; j++)
                {
                    //set weights to random values
                    //wHiddenOutput[j][i] = (distribution(generator)) * limit;
                    //setElem(j, i, nHidden + 1, wHiddenOutput, (distribution(generator)) * limit);
                    setWHO(j, i, (distribution(generator)) * limit);
                }
            }
        }
};

namespace trainerNoMomentum{
	
}

