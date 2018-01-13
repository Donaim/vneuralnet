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

            ErrorGradients = new double[nn.size.Count - 1][]; // same as numbers of neurons - 1 input neuron
            for(int i = 0; i < ErrorGradients.Length; i++) {
                ErrorGradients[i] = new double[nn.size[i + 1]]; // next error cares, so i + 1; Cannot think of error of input neurons
            }
        }

        public void backpropagate(double[] desiredOutputs, double learningRate){
            int i = nn.N.Length - 1; // starting from last neuron layer

            // trainerNoMomentum.getOEG(nn.N[i], desired:desiredOutputs, gradient:ErrorGradients[i]);
            // trainerNoMomentum.mult(nn.L[i - 1], nn.N[i - 1], ErrorGradients[i], learningRate: learningRate, insize: nn.size[i - 1], outsize: nn.size[i]);
            // i--;

            // while(i >= 1) {
            //     trainerNoMomentum.getHEG(nn.L[i], nn.N[i], ErrorGradients[i + 1], ErrorGradients[i], nn.size[i + 1], nn.size[i]);
            //     trainerNoMomentum.mult(nn.L[i - 1], nn.N[i - 1], ErrorGradients[i], learningRate: learningRate, insize: nn.size[i - 1], outsize: nn.size[i]);

            //     i--;
            // }


            // getOEG(NN.outputNeurons, desiredOutputs, outputErrorGradients);
            // mult(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, learningRate, NN.nHidden, NN.nOutput);

            // getHEG(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, hiddenErrorGradients, NN.NOutput, NN.nHidden);
            // mult(NN.wInputHidden, NN.inputNeurons, hiddenErrorGradients, learningRate, NN.nInput, NN.NHidden);
        }

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void TrainOne(double[] inputs, double[] desiredOutputs, double learningRate)
        {
            nn.feedForward(inputs);
            backpropagate(desiredOutputs, learningRate);
        }
    }
}