using System;
// using Microsoft.VisualStudio.TestTools.UnitTesting;

using System.Linq;
using System.IO;
using static System.Console;

using VNNLib;
using VNNAddOn;

namespace StdTest
{
    // [TestClass]
    public class kerastest
    {
        // [TestMethod]
        // [TestCategory("KEK")]
        public void keras_vnnCM()
        {
            var nn = new vnnCm(
               new double[][,] 
               {
                   new double[,] { { 0.5, 0.5, 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5, 0.5, 0.5 } },
                   new double[,] { { 0.3, 0.3 }, { 0.3, 0.3 }, { 0.3, 0.3 }, { 0.3, 0.3 }, { 0.3, 0.3 }, },
               },
               new double[][] 
               {
                   new double[] { 0.1, 0.1, 0.1, 0.1, 0.1 },
                   new double[] { 0.5, 0.5 },
               },
               (x) => x
               );

            predict(nn, 1, 1, 1, 1);
            predict(nn, 1, 1, 1, 0);
            predict(nn, 4, 0, 4, 4);
            predict(nn, 4, 3, 2, 4);
        }

        // [TestMethod]
        // [TestCategory("lol")]
        public void keras_save_load()
        {
            var nn = vnnCm.LoadTxt(@"d:\keras_save_load\");
            predict(nn, 0.7, 0.7);
            predict(nn, 0.7, -0.7);
            predict(nn, -0.7, 0.7);
            predict(nn, -0.7, -0.7);
        }

        static void predict(vnnCm nn, params double[] inp)
        {
            var o = nn.feedResult(inp);
            WriteLine($"({string.Join("  ", inp.Select(z => z.ToString("N2")))}) -> ({string.Join("  ", o.Select(z => z.ToString("N2")))})");
        }
    }
}
