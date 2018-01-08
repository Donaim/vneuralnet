using System;
// using Microsoft.VisualStudio.TestTools.UnitTesting;

using System.Linq;
using System.IO;
using static System.Console;

using VNNLib;
using VNNAddOn;

using vutils.Testing;

namespace StdTest
{
    // [TestClass]
    public class vnnCppTest
    {
        // [TestMethod]
        [TestingObject]
        public unsafe void CreateTest()
        {
            var nn = new vnnCpp(10, 2000, 10);
            //for (int i = 0; i < 5; i++) { nn.wInputHidden[i][0] = -1; }

            nn.Print();
        }
    }
}
