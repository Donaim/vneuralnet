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

namespace StdTest
{
    class SwappingWeights
    {
        const string savepath = @"d:\serv\nntmp2";
        const int nI = 10, nH = 1111, nO = 5;

        // [TestMethod]
        void saveRandomWeights()
        {
            //var nn = new vnn(nI, nH, nO, (mlp) => VNNAddOn.addon.RandomizeUniform(mlp));

            //printAnswer(nn);

            //File.WriteAllBytes(savepath, nn.ToBytes());
            var nn2 = new vnn(File.ReadAllBytes(savepath));
            //var nn2 = new vnn(nI, nH, nO, (mlp) => VNNAddOn.addon.RandomizeUniform(mlp));

            printAnswer(nn2);
        }
        static void printAnswer(IFeedResultNN nn)
        {
            double[] first = new double[nI] { 0.1, -0.7, 0.3, 0.99, -0.2, -0.4, 0.3, 0.1, -0.9, -0.2 };
            WriteLine(string.Join(" ", nn.feedResult(first).Select(o => o.ToString("N2"))));
        }
    }
}
