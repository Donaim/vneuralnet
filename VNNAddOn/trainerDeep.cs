using System.Runtime.CompilerServices;

using static System.Math;
using VNNLib;


namespace VNNAddOn
{
    public class trainerDeep : ITrainer { // No momentum
        public readonly vnnDeep nn;
        readonly double[][] ErrorGradients;

        public trainerDeep(vnnDeep _nn) {
            nn = _nn;

            ErrorGradients = new double[nn.N.Length][]; // same as numbers of neurons
            for(int i = 0; i < ErrorGradients.Length; i++) {
                ErrorGradients[i] = new double[nn.N[i].Length];
            }
        }

        public void backpropagate(double[] desiredOutputs, double learningRate){
            int i = nn.N.Length - 1; // starting from last neuron layer
            int wi = nn.L.Length - 1;

            trainerNoMomentum.getOEG(nn.N[i], desired:desiredOutputs, gradient: ErrorGradients[i]);
            // trainerNoMomentum.mult(nn.L[wi], nn.N[i - 1], ErrorGradients[i], learningRate: learningRate);
            i--;
            wi--;

            // while(i >= 1) {
            //     trainerNoMomentum.getHEG(nn.L[wi], nn.N[i], ErrorGradients[i + 1], ErrorGradients[i]);
            //     trainerNoMomentum.mult(nn.L[wi - 1], nn.N[i - 1], ErrorGradients[i], learningRate);

            //     i--;
            //     wi--;
            // }


            // getOEG(NN.outputNeurons, desiredOutputs, outputErrorGradients);
            // mult(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, learningRate);

            // getHEG(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, hiddenErrorGradients, NN.NOutput, NN.nHidden);
            // mult(NN.wInputHidden, NN.inputNeurons, hiddenErrorGradients, learningRate);
       }

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void TrainOne(double[] inputs, double[] desiredOutputs, double learningRate)
        {
            nn.feedForward(inputs);
            backpropagate(desiredOutputs, learningRate);
        }
    }
}