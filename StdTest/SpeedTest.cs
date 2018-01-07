using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

// using Microsoft.VisualStudio.TestTools.UnitTesting;

using System.IO;
using static System.Console;

using VNNLib;
using VNNAddOn;

namespace StdTest
{
    // [TestClass]
    public class SpeedTest
    {
        public const int INPUT_SIZE = 200, HIDDEN_SIZE = 1000, OUTPUT_SIZE = 100;

        // [TestMethod]
        public void SimpleVNNForwardSpeedTest()
        {
            var nn = new vnn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, addon.RandomizeWeights);
            WriteLine($"Time: {measureForwardSpeed(nn)}");
        }
        // [TestMethod]
        static void SimpleVNNBackpropNoMomentumSpeedTest()
        {
            var nn = new vnn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, (mlp) => addon.RandomizeUniform(mlp));
            var tr = new trainerNoMomentum(nn);
            WriteLine($"Time: {measureBackpropSpeed(nn, tr)}");
        }


        // [TestMethod]
        public void VNNCPPForwardSpeedTest()
        {
            //throw new NotImplementedException();
            var nn = new vnnCpp(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            WriteLine($"Time: {measureForwardSpeed(nn)}");
        }

        const double LEARNING_RATE = 0.01;
        static double measureBackpropSpeed(IFeedForwardNN nn, ITrainer tr)
        {
            for (int i = 0; i < 20; i++) { nn.feedForward(GetRandomInputs(INPUT_SIZE)); }

            var sw = Stopwatch.StartNew();

            const int laps = 3000;
            //const int laps = 1000;
            for (int i = 0; i < laps; i++)
            {
                nn.feedForward(GetRandomInputs(INPUT_SIZE));
                tr.backpropagate(GetRandomInputs(OUTPUT_SIZE), LEARNING_RATE);
            }

            return sw.ElapsedMilliseconds;
        }
        static double measureForwardSpeed(IFeedForwardNN nn)
        {
            for(int i = 0; i < 100; i++) { nn.feedForward(GetRandomInputs(INPUT_SIZE)); }

            var sw = Stopwatch.StartNew();

            const int laps = 10000;
            //const int laps = 1000;
            for(int i = 0; i < laps; i++)
            {
                nn.feedForward(GetRandomInputs(INPUT_SIZE));
            }

            return sw.ElapsedMilliseconds;
        }

        static SpeedTest()
        {
            var buffertmp = new double[BUFFER_SIZE];

            Random r = new Random();
            for(int i = 0; i < BUFFER_SIZE; i++) { buffertmp[i] = r.NextDouble() * 2 - 1; }

            buffer = buffertmp;
        }
        static readonly IReadOnlyList<double> buffer;
        static ulong index = 1;
        public const int BUFFER_SIZE = INPUT_SIZE * 10 + 7;
        public static double[] GetRandomInputs(int size)
        {
            var re = new double[size];

            for (int i = 0; i < size; i++, index++)
            {
                re[i] = buffer[(int)(index % BUFFER_SIZE)];
            }

            return re;
        }
    }
}
