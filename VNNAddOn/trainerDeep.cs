using System.Runtime.CompilerServices;

using static System.Math;
using VNNLib;


namespace VNNAddOn
{
    public class trainerDeep { // No momentum
        public readonly vnnDeep nn;
        readonly double[][] ErrorGradients;

        public trainerDeep(vnnDeep _nn) {
            nn = _nn;

            // ErrorGradients = new double[nn.]
        }
    }
}