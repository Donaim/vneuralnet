using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using static System.Math;

using System.Runtime.InteropServices;

namespace VNNLib
{
    public sealed class vnnCpp : IFeedResultNN, IFeedForwardNN
    {
        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void* Create(int nInputs, int nHidden, int nOutputs);
        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void Hello(void* p);

        //[DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        //public unsafe static extern void StaticHello();

        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void feedForward(void* p, double[] pattern);

        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void printWeights(void* p);

        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* get_wInputHidden(void* p);
        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* get_wHiddenOutput(void* p);

        [DllImport(@"vnnCppEngine.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* get_outputNeurons(void* p);

        public unsafe void feedForward(double[] pattern) => feedForward(ptr, pattern);
        public unsafe double[] feedResult(double[] pattern)
        {
            feedForward(pattern);

            //create copy of output results
            double[] results = new double[nOutput];
            for (int i = 0; i < nOutput; i++) { results[i] = outputNeurons[i]; }

            return results;
        }

        public readonly int nInput, nHidden, nOutput;
        public readonly unsafe double* wInputHidden, wHiddenOutput;
        public readonly unsafe double* outputNeurons;
        readonly unsafe void* ptr;
        public unsafe vnnCpp(int _nInputs, int _nHidden, int _nOutput)
        {
            nInput = _nInputs; nHidden = _nHidden; nOutput = _nOutput;

            ptr = Create(nInput, nHidden, nOutput);

            wInputHidden = get_wInputHidden(ptr);
            wHiddenOutput = get_wHiddenOutput(ptr);
            outputNeurons = get_outputNeurons(ptr);

            //Hello(ptr);
        }

        public unsafe void Print()
        {
            printWeights(ptr);
        }
    }
}
