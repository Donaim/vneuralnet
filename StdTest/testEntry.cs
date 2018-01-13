using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StdTest
{
    class TestEntry
    {
        static void Main(string[] args)
        {
            //new vnnCppTest().CreateTest();
            //new SpeedTest().VNNCPPForwardSpeedTest();
            //new SpeedTest().OptimizedVNNForwardSpeedTest();
            //new WeightsTest().TestUniformRandomization();

            vutils.Testing.TestingModule.ChooseMethodsLoop(false);
            //Console.ReadLine();
        }
    }
}
