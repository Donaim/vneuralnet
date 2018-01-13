
using vutils.Testing;
using vutils.Plotting;

using static System.Console;

using System.Threading;

using VNNLib;
using VNNAddOn;

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
        // WriteLine("HELLO");
        var deep = vnnDeep.CreateRandom(rand, 2, 3, 3, 4, 5, 3);
        WriteLine(deep.ToString(true)); 
        deep.feedForward(getRandVector(2));
        WriteLine(deep.ToString(true)); 



    }
}
