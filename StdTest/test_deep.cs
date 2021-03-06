using System.Linq;
using vutils.Testing;
using vutils.Plotting;

using static System.Console;

using System.Threading;

using VNNLib;
using VNNAddOn;
using System.Diagnostics;
using System;

class test_deep {

    public double rand(int n_in, int n_out, int i) => addon.RandomizeUniform(n_in, n_out);
    public double[] getRandVector(int len) {
        var inputs = new double[len];
        for(int i = 0; i < len; i++) { inputs[i] = 1; }
        for (int i = 0; i < len / 10; i++) { inputs[i] = rand(len, len, i); }
        return inputs;
    }

    [TestingObject]
    public void test_deep_first(){
        addon.RSeed = 100;                

        var deep = vnnDeep.CreateRandom(rand, 2, 3, 3, 4, 5, 3);
        WriteLine(deep.ToString(true)); 
        deep.feedForward(getRandVector(2));
        WriteLine(deep.ToString(true)); 
    }
    [TestingObject]
    public void test_deep_save(){
        var deep = vnnDeep.CreateRandom(rand, 2, 3, 3, 4, 5, 3);
        WriteLine(deep.ToString(false)); 
        // WriteLine(string.Join(",", deep.size));
        // WriteLine(string.Join(",", deep.L.Select(o => o.GetLength(0))));
        // WriteLine(string.Join(",", deep.L.Select(o => o.GetLength(1))));

        var bytes = deep.ToBytes();

        var recover = new vnnDeep(bytes);
        WriteLine(recover.ToString(false));
    }

    [TestingObject]
    public void test_deep_trainer(){
        var nn = vnnDeep.CreateRandom(rand, 2, 30, 20, 20, 20, 2);
        var tr = new trainerDeep(nn);
        var tset = data.DataSets.twoParamTest;

        Stopwatch sw = Stopwatch.StartNew();
        const double ACCRATE = 0.05;

        for (int i = 0; i < 500; i++)
        {
            for(int j = 0, to = tset.size; j < to; j++)
            {
                tr.TrainOne(tset.inputs[j], tset.outputs[j], 0.1);
            }
        }
        var acc = tset.getAccuracy(nn, ACCRATE);
        // if (acc < 0.9) { ($"Not enought accuracy: {acc.ToString("N2")}"); }

        WriteLine("Elapsed = " + sw.ElapsedMilliseconds + " ms.");
        WriteLine($"Accuracy (+-{ACCRATE}) = {acc.ToString("N2")}");
        WriteLine("Random MSE = " + tset.getRandomMSE(nn, 500));
    }

    [TestingObject]
    public void test_deep_sizes(){
        var nnd = vnnDeep.CreateRandom(rand, 2, 100, 2);
        var nn = new vnn(2, 100, 2, (mlp) => addon.RandomizeUniform(mlp));

        Console.WriteLine($"nn= wih:{nn.wInputHidden.GetLength(0)}x{nn.wInputHidden.GetLength(1)} who:{nn.wHiddenOutput.GetLength(0)}x{nn.wHiddenOutput.GetLength(1)}");
        Console.WriteLine($"nd= wih:{nnd.L[0].GetLength(0)}x{nnd.L[0].GetLength(1)} who:{nnd.L[1].GetLength(0)}x{nnd.L[1].GetLength(1)}");
        Console.WriteLine($"nn= nin:{nn.inputNeurons.Length} nh:{nn.hiddenNeurons.Length} no:{nn.outputNeurons.Length}");
        Console.WriteLine($"nd= nin:{nnd.N[0].Length} nh:{nnd.N[1].Length} no:{nnd.N[2].Length}");
        Console.WriteLine($"nd= sin:{nnd.size[0]} sh:{nnd.size[1]} so:{nnd.size[2]}");
        Console.WriteLine($"nd= size:{nnd.size.Count}");
    }

    [TestingObject]
    public void test_deep_conversion(){
        var nnd = vnnDeep.CreateRandom(rand, 2, 100, 2);
        var nn = nnd.ToSimpleVNN();
        addon.RandomizeUniform(nn);

        StdTest.old_test.simple_portable(nn);
    }

    [TestingObject]
    public void test_deep_outputs(){
        var nnd = vnnDeep.CreateRandom(rand, 2, 3, 3);
        var nn = nnd.ToSimpleVNN();
        var tset = data.DataSets.twoParamTest;

        const bool neurons = false;

        WriteLine(nnd.ToString(neurons));
        WriteLine(nn.ToString(neurons));
        
        nnd.feedResult(tset.inputs[0]);
        nn.feedResult(tset.inputs[0]);

        WriteLine(nnd.ToString(neurons));
        WriteLine(nn.ToString(neurons));
    }
}
