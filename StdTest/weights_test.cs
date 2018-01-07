using System;
// using Microsoft.VisualStudio.TestTools.UnitTesting;

using System.Linq;
using System.IO;
using static System.Console;

using vutils.Statistics;
using vutils.Plotting;

using VNNLib;
using VNNAddOn;

namespace StdTest
{
    // [TestClass]
    public class WeightsTest
    {
        static readonly Random R = new Random();

        // [TestMethod]
        public void TestStdRandomization()
        {
            testZeroInputs(addon.RandomizeWeights);
            testOnesInputs(addon.RandomizeWeights);
            testRandomInputs(addon.RandomizeWeights);
        }
        // [TestMethod]
        public void TestUniformRandomization()
        {
            testZeroInputs((nn) => addon.RandomizeUniform(nn, 5.0, 3.5));
            testOnesInputs((nn) => addon.RandomizeUniform(nn, 1.5, 3.5));
            testRandomInputs((nn) => addon.RandomizeUniform(nn, 3, 3.5));
        }

        void testOnesInputs(Action<ISimpleMLP> rand)
        {
            WriteLine("#Ones inputs");
            var inputs = new double[NINPUTS];
            for(int i = 0; i < NINPUTS; i++) { inputs[i] = 1; }
            for (int i = 0; i < NINPUTS / 10; i++) { inputs[R.Next(NINPUTS)] = 0; }
            testWeights(inputs, rand, "ones inputs");
        }
        void testZeroInputs(Action<ISimpleMLP> rand)
        {
            WriteLine("#Zero inputs");
            var inputs = new double[NINPUTS];
            for (int i = 0; i < NINPUTS / 10; i++) { inputs[R.Next(NINPUTS)] = 1; }
            testWeights(inputs, rand, "zero inputs");
        }
        void testRandomInputs(Action<ISimpleMLP> rand)
        {
            WriteLine("#Rand inputs");
            var inputs = new double[NINPUTS];
            for (int i = 0; i < NINPUTS; i++) { inputs[i] = R.NextDouble(); }
            testWeights(inputs, rand, "rand inputs");
        }

        const int NINPUTS = 72, NHIDDEN = 100, NOUTPUT = 1000;
        unsafe void testWeights(double[] inputs, Action<ISimpleMLP> randFunc, string name)
        {
            var nn = new vnn(NINPUTS, NHIDDEN, NOUTPUT, randFunc);
            //var nn = new vnnCpp(NINPUTS, NHIDDEN, NOUTPUT);

            var re = nn.feedResult(inputs);

            //Histogram.PrintHist(nn.hiddenNeurons);
            Histogram.PrintHist(re);
            Histogram.ShowHist(re, name: name);
            //PrintHist(re);

            //Write("Inputs\t: ");
            //PrintHist(nn.inputNeurons.Take(NINPUTS).ToArray());
            //Write("Hidden\t: ");
            //PrintHist(nn.hiddenNeurons.Take(NHIDDEN).ToArray(), lower: 0, upper: 1);
            //Write("Output\t: ");
            //PrintHist(nn.outputNeurons, lower: 0, upper: 1);
        }
    }
}
