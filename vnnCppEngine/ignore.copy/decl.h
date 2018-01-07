int nInput; 
int nHidden;
int nOutput;
double* inputNeurons;
double* hiddenNeurons;
double* outputNeurons;
double** wInputHidden;
double** wHiddenOutput;

void HelloWorld();
void printWeights();

double** createEmptyArr(int in, int out);

void CreateNN2(int _nInputs, int _nHidden, int _nOutputs, double** _wInputHidden, double** _wHiddenOutput);
void CreateNN(int _nInputs, int _nHidden, int _nOutputs);

void feedForward(double* pattern);

void RandomizeUniform(double mult1 = 5, double mult2 = 3.5);
