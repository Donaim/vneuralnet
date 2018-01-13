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

            trainerNoMomentum.getOEG(nn.N[i], desired:desiredOutputs, gradient: ErrorGradients[i]);
            trainerNoMomentum.mult(nn.L[i - 1], nn.N[i], ErrorGradients[i], learningRate: learningRate);
            i--;
          

            // while(i >= 1) {
            //     trainerNoMomentum.getHEG(nn.L[i], nn.N[i], ErrorGradients[i + 1], ErrorGradients[i], nn.size[i + 1], nn.size[i]);
            //     trainerNoMomentum.mult(nn.L[i - 1], nn.N[i - 1], ErrorGradients[i], learningRate: learningRate, insize: nn.size[i - 1], outsize: nn.size[i]);

            //     i--;
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