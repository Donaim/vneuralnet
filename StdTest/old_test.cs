using System;
using System.Text;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;

// using Microsoft.VisualStudio.TestTools.UnitTesting;
using vutils.Testing;

using VNNAddOn;
using VNNLib;
using static System.Console;
using static System.Math;

namespace StdTest
{
    // [TestClass]
    public class old_test
    {
        public old_test()
        {

        }
        
        // [TestMethod]
        [TestingObject]
        public void simple()
        {
            vnn nn = new vnn(2, 100, 2, (mlp) => addon.RandomizeUniform(mlp));
            simple_portable(nn);
        }
        public static void simple_portable(vnn nn)
        {
            //var tr = new trainerModern(nn);
            var tr = new trainerNoMomentum(nn);
            var tset = data.DataSets.twoParamTest;

            Stopwatch sw = Stopwatch.StartNew();
            const double ACCRATE = 0.05;

            for (int i = 0; i < 500; i++)
            {
                //tr.TrainEpoch(tset.inputs, tset.outputs, 0.1, 0.9);
                for(int j = 0, to = tset.size; j < to; j++)
                {
                    tr.TrainOne(tset.inputs[j], tset.outputs[j], 0.1);
                }
            }
            var acc = tset.getAccuracy(nn, ACCRATE);
            // if (acc < 0.9) { throw new Exception($"Not enought accuracy: {acc.ToString("N2")}"); }

            WriteLine("Elapsed = " + sw.ElapsedMilliseconds + " ms.");
            WriteLine($"Accuracy (+-{ACCRATE}) = {acc.ToString("N2")}");
            WriteLine("Random MSE = " + tset.getRandomMSE(nn, 500));
        }

        public void straight()
        {
            int size = 25;

            vnn nn = new vnn(size, 500, size, addon.RandomizeWeights);
            var tr = new trainerModern(nn);
            List<double[]> history = new List<double[]>();
            trainingSet tset;

            Stopwatch sw = Stopwatch.StartNew();

            int counter = 0;
            var rand = new Random();
            do
            {
                var data = new double[size];
                data[rand.Next(0, data.Length)] = 1.0;
                history.Add(data);
                tset = new trainingSet(history.ToArray(), history.ToArray());

                tr.TrainEpoch(tset.inputs, tset.outputs, 0.02, 0.9);
                //tr.TrainOne(data, data, 0.1, 0.9);

                counter++;
                Console.Title = $"{counter} ({counter * tset.size})";
            }
            while (tset.getAccuracy(nn, 0.1) < 0.9);

            WriteLine("Elapsed = " + sw.ElapsedMilliseconds + " ms.");
            WriteLine("MSE = " + tset.getMSE(nn));
            WriteLine("Random MSE = " + tset.getRandomMSE(nn, 500));

            nn.TestLoop(new Func<double[], double[]>(arr => new double[] { arr[0] > 0.5 ? 1 : 0, Sin(arr[0] + arr[1]) * arr[0] }));
        }
        public void bounded_substraction()
        {
            /// <summary>
            /// WORKS!!!! :)
            /// Testing if hearthstone health and atack should be one in one input neuron or in whole 'one-hot' input vector
            /// </summary>
            vnn nn = new vnn(2, 100, 1, addon.RandomizeWeights);
            var tr = new trainerModern(nn);
            var tset = getSet(1000);

            Stopwatch sw = Stopwatch.StartNew();

            int counter = 0;
            do
            {
                tr.TrainEpoch(tset.inputs, tset.outputs, 0.1, 0.9);

                counter++;
                Console.Title = $"{counter} ({counter * tset.size})";
            }
            while (tset.getAccuracy(nn, 0.05) < 1);

            WriteLine("Elapsed = " + sw.ElapsedMilliseconds + " ms.");
            WriteLine("MSE = " + tset.getMSE(nn));

            nn.TestLoop(input_translator, output_translator);
            //nn.TestLoop();

            const int maxVal = 10;
            string output_translator(double[] arr)
            {
                int output = (int)Round(arr[0] * maxVal);
                return output.ToString();
            }
            double[] input_translator(string text)
            {
                string[] split = text.Split(' ');

                int a = int.Parse(split[0]);
                double da = a / (double)maxVal;

                int b = int.Parse(split[1]);
                double db = b / (double)maxVal;

                return new double[] { da, db };
            }
            trainingSet getSet(int size)
            {
                var rand = new Random();
                double[][] inp = new double[size][], outp = new double[size][];

                for (int i = 0; i < size; i++)
                {
                    int a = rand.Next(0, maxVal);
                    double da = a / (double)maxVal;

                    int b = rand.Next(0, maxVal);
                    double db = b / (double)maxVal;

                    int answer = Max(0, a - b);
                    double dansw = answer / (double)maxVal;

                    inp[i] = new double[] { da, db };
                    outp[i] = new double[] { dansw };
                }

                return new trainingSet(inp, outp);
            }
        }

        public void test_binary()
        {
            vnn nn = new vnn(2, 5, 2);
            reporter rep = new reporter(nn);
            Console.WriteLine(rep.wHiddenOutput);

            byte[] bts = nn.ToBytes();
            vnn newnn = new vnn(bts);

            rep = new reporter(newnn);
            Console.WriteLine(rep.wHiddenOutput);

            Console.ReadKey();
        }

        class primes
        {
            public primes()
            {
                Console.WriteLine(get(100));
                Console.ReadKey();
            }
            void gen(int n)
            {
                uint x = 0;
                using (var stream = new System.IO.FileStream(@"E:\OneDrive\Other\ApplicationServer\Data\uintBinaryPrimes", System.IO.FileMode.Create))
                {
                    using (var writer = new System.IO.BinaryWriter(stream))
                    {
                        for (int i = 0; i < n; i++, x++)
                        {
                            while (!IsPrime(x)) { x++; }
                            writer.Write(x);
                        }
                    }
                }
            }
            uint get(int i)
            {
                using (var stream = new System.IO.FileStream(@"E:\OneDrive\Other\ApplicationServer\Data\uintBinaryPrimes", System.IO.FileMode.Open))
                {
                    using (var reader = new System.IO.BinaryReader(stream))
                    {
                        stream.Seek(i * sizeof(uint), System.IO.SeekOrigin.Begin);
                        return reader.ReadUInt32();
                    }
                }
            }

            public static bool IsPrime(uint x)
            {
                if (x == 2) return true;
                Random r = new Random();
                for (int i = 0; i < 100; i++)
                {
                    uint a = ((uint)r.Next() % (x - 2)) + 2;
                    if (GCD(a, x) != 1)
                        return false;
                    if (pows(a, x - 1, x) != 1)
                        return false;
                }
                return true;
            }
            public static uint GCD(uint a, uint b)
            {
                if (b == 0)
                    return a;
                return GCD(b, a % b);
            }
            static uint mul(uint a, uint b, uint m)
            {
                if (b == 1)
                    return a;
                if (b % 2 == 0)
                {
                    uint t = mul(a, b / 2, m);
                    return (2 * t) % m;
                }
                return (mul(a, b - 1, m) + a) % m;
            }
            static uint pows(uint a, uint b, uint m)
            {
                if (b == 0)
                    return 1;
                if (b % 2 == 0)
                {
                    uint t = pows(a, b / 2, m);
                    return mul(t, t, m) % m;
                }
                return (mul(pows(a, b - 1, m), a, m)) % m;
            }
        }
        static void test_randomizing()
        {
            vnn nn;
            trainer tr;
            double mean = 0.0;
            double to = 1.0;
            for (int i = 0; i < to; i++)
            {
                nn = new vnn(2, 2, 1);
                tr = new trainer(nn, data.DataSets.xorProblem, 0.1, 0.9);
                Stopwatch sw = Stopwatch.StartNew();
                tr.TrainUntilAccuracy(0.8, 0.2, 100);
                mean += sw.ElapsedMilliseconds;
            }
            Console.WriteLine(mean / to);
            Console.ReadKey();
        }
        class dynamicSynapsys
        {
            public dynamicSynapsys()
            {
                //test();
                newtraintertest();
            }
            void newtraintertest()
            {
                vnn nn = new vnn(2, 100, 2, addon.RandomizeWeights);
                var tr = new data.CustomClasses.randomUpdateTrainer(nn, data.DataSets.twoParamTest, 0.1, 0.9);

                Stopwatch sw = Stopwatch.StartNew();
                //tr.TrainUntilMSE(0.01, 10);
                tr.TrainUntilAccuracy(0.9, 0.1);
                WriteLine("Elapsed = " + sw.ElapsedMilliseconds);
                WriteLine("MSE = " + tr.getMSE());
            }
            void test()
            {
                vnn nn = new vnn(2, 1000, 2, addon.RandomizeWeights);
                trainer tr = new trainer(nn, data.DataSets.twoParamTest, 0.01, 0.9);
                tr.TrainUntilAccuracy(0.9, 0.1);

                int counter = 0;
                double mean = 0.0;
                foreach (var v in nn.wHiddenOutput)
                {
                    mean += Abs(v);
                    if (Abs(v) < 1.0 / nn.nHidden) { counter++; }
                }
                mean = mean / nn.wHiddenOutput.Length;

                WriteLine("weak = " + counter);
                WriteLine("% = " + counter / (double)nn.wHiddenOutput.Length);
                WriteLine("Mean = " + mean + "; level = " + 1.0 / nn.nHidden);
            }
        }
        class fullDynamics
        {
            public fullDynamics()
            {
                var nn =
                    //new vnn(2, 100, 2);
                    new data.CustomClasses.DynamicSynapsis.DynamicVNN(2, 100, 2, 0.1);
                var tr =
                    //new trainer(nn, data.DataSets.twoParamTest, 0.005, 0.9);
                    new data.CustomClasses.DynamicSynapsis.DynamicTrainer(nn, data.DataSets.twoParamTest, 0.1, 0.9);
                WriteLine("Dynamic objects created");
                WriteLine("LiveRatio Begining = " + (nn.aInputHidden + nn.aHiddenOutput));
                WriteLine("                 % = " + (nn.aInputHidden + nn.aHiddenOutput) / (double)(((nn.nInput + 1) * nn.nHidden + (nn.nHidden + 1) * nn.nOutput)));

                Stopwatch sw = Stopwatch.StartNew();
                //tr.TrainFor(100000);
                tr.TrainDynamicallyFor(100000);
                WriteLine("LiveRatio End = " + (nn.aInputHidden + nn.aHiddenOutput));
                WriteLine("            % = " + (nn.aInputHidden + nn.aHiddenOutput) / (double)(((nn.nInput + 1) * nn.nHidden + (nn.nHidden + 1) * nn.nOutput)));

                WriteLine("Elapsed = " + sw.ElapsedMilliseconds);
                WriteLine("MSE = " + tr.getMSE());
                WriteLine("Acc = " + tr.getAccuracy(0.1));
            }
        }
        class sortingTest
        {
            public sortingTest()
            {
                int n = 3;

                var nn = new vnn(n, 100, n);
                var tr = new trainer(nn, data.DataSets.sorting(n), 0.1, 0.9);

                tr.TrainFor(10000);
                System.IO.File.WriteAllBytes("E:\\OneDrive\\Other\\Temp\\tempNN", nn.ToBytes());
                WriteLine("MSE = " + tr.getMSE());
                WriteLine("Acc = " + tr.getAccuracy(1 / (double)n));
                nn.TestLoop();
            }
        }
    }
}
