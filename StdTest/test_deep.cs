
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
        var bytes = deep.ToBytes();

        var recover = new vnnDeep(bytes);
        WriteLine(recover.ToString(false));
    }

    [TestingObject]
    public void test_deep_trainer(){
        var nn = vnnDeep.CreateRandom(rand, 2, 200, 2);
        var tr = new trainerDeep(nn);
        var tset = data.DataSets.twoParamTest;

        Stopwatch sw = Stopwatch.StartNew();
        const double ACCRATE = 1;
        // const double ACCRATE = 0.05;

        for (int i = 0; i < 500; i++)
        {
            //tr.TrainEpoch(tset.inputs, tset.outputs, 0.1, 0.9);
            for(int j = 0, to = tset.size; j < to; j++)
            {
                tr.TrainOne(tset.inputs[j], tset.outputs[j], 0.1);
            }
        }
        var acc = tset.getAccuracy(nn, ACCRATE);
        if (acc < 0.9) { throw new Exception($"Not enought accuracy: {acc.ToString("N2")}"); }

        WriteLine("Elapsed = " + sw.ElapsedMilliseconds + " ms.");
        WriteLine($"Accuracy (+-{ACCRATE}) = {acc.ToString("N2")}");
        WriteLine("Random MSE = " + tset.getRandomMSE(nn, 500));
    }
}
