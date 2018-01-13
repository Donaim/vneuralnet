using System.Runtime.CompilerServices;

using static System.Math;
using VNNLib;

namespace VNNAddOn
{
	public class trainer
	{
		public readonly vnn NN;
		public readonly trainingSet tset;
		public trainer(vnn net, trainingSet set, double learning_rate, double moment)
		{
			exceptions.NNSetNotMatch.check_ex(net, set);
			NN = net;
			tset = set;

			momentum = moment;
			learningRate = learning_rate;

			outputErrorGradients = new double[net.nOutput];
			hiddenErrorGradients = new double[net.nHidden];

			deltaHiddenOutput = new double[net.nOutput, net.nHidden + 1];
            deltaInputHidden = new double[net.nHidden, net.nInput + 1];
		}

		public void TrainUntilAccuracy(double desired_accuracy, double desired_precision, int check_interval = 1000)
		{
			while(getAccuracy(desired_precision) < desired_accuracy)
			{
				for(int i = 0; i < check_interval; i++)
				{
					NN.feedForward(tset.inputs[i % tset.size]);
					backpropagate(tset.outputs[i % tset.size]);
				}
			}
		}
		public void TrainUntilMSE(double desired_MSE, int check_interval = 1000)
		{
			while(getMSE() > desired_MSE)
			{
				for(int i = 0; i < check_interval; i++)
				{
					NN.feedForward(tset.inputs[i % tset.size]);
					backpropagate(tset.outputs[i % tset.size]);
				}
			}
		}
		public void TrainFor(int n)
		{
			for(int i = 0; i < n; i++)
			{
				//vmon.Line("inputs = {" + string.Join(", ", tset.inputs[i % tset.size]) + '}' + "(size = " + (i % tset.size) + ")");
				NN.feedForward(tset.inputs[i % tset.size]);
				backpropagate(tset.outputs[i % tset.size]);
			}
		}
		public void TrainEpoch()
		{
			for(int i = 0; i < tset.size; i++)
			{
				NN.feedForward(tset.inputs[i]);
				backpropagate(tset.outputs[i]);
			}
		}
		
		public double learningRate, momentum;
		protected double[] outputErrorGradients, hiddenErrorGradients;
		protected double[,] deltaHiddenOutput, deltaInputHidden;
		public virtual unsafe void backpropagate(double[] desiredOutputs)
		{
            //modify deltas between hidden and output layers
            //--------------------------------------------------------------------------------------------------------
            for (int k = 0; k < NN.nOutput; k++)
            {
                //get error gradient for every output node
                outputErrorGradients[k] = NN.outputNeurons[k] * (1 - NN.outputNeurons[k]) * (desiredOutputs[k] - NN.outputNeurons[k]);
                //for all nodes in hidden layer and bias neuron
                for (int j = 0; j <= NN.nHidden; j++)
                {
                    //calculate change in weight
                    deltaHiddenOutput[k, j] = learningRate * NN.hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[k, j];
                    //vmon.Line("[H] neuron value = " + NN.hiddenNeurons[j] + "; neuron delta = " + deltaHiddenOutput[j, k] + "; error gradient = " + outputErrorGradients[k]);
                }
            }

            //modify deltas between input and hidden layers
            //--------------------------------------------------------------------------------------------------------
            for (int j = 0; j < NN.nHidden; j++)
            {
                //get error gradient for every hidden node
                double weightedSum = 0;
                for (int k = 0; k < NN.nOutput; k++) weightedSum += NN.wHiddenOutput[k, j] * outputErrorGradients[k];
                hiddenErrorGradients[j] = NN.hiddenNeurons[j] * (1 - NN.hiddenNeurons[j]) * weightedSum;

                //for all nodes in input layer and bias neuron
                for (int i = 0; i <= NN.nInput; i++)
                {
                    //calculate change in weight 
                    deltaInputHidden[j, i] = learningRate * NN.inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[j, i];
                    //vmon.Line("[I] neuron value = " + NN.inputNeurons[i] + "; neuron delta = " + deltaInputHidden[i, j] + "; error gradient = " + hiddenErrorGradients[j]);
                }
            }
            //vmon.Line("");

            //if using stochastic learning update the weights immediately
            updateWeights();
		}
		protected virtual void updateWeights()
		{
			//input -> hidden weights
			//--------------------------------------------------------------------------------------------------------
			for(int i = 0; i <= NN.nInput; i++)
			{
				for(int j = 0; j < NN.nHidden; j++)
				{
					//update weight
					NN.wInputHidden[j, i] += deltaInputHidden[j, i];
				}
			}

			//hidden -> output weights
			//--------------------------------------------------------------------------------------------------------
			for(int j = 0; j <= NN.nHidden; j++)
			{
				for(int k = 0; k < NN.nOutput; k++)
				{
					//update weight
					NN.wHiddenOutput[k, j] += deltaHiddenOutput[k, j];
				}
			}
		}
	
		public unsafe double getMSE()
		{
			double mse = 0;

			//for every training input array
			for(int tp = 0, l = tset.size; tp < l; tp++)
			{
				//feed inputs through network and backpropagate errors
				NN.feedForward(tset.inputs[tp]);

				//check all outputs against desired output values
				for(int k = 0; k < NN.nOutput; k++)
				{
					//sum all the MSEs together
					mse += Abs(NN.outputNeurons[k] - tset.outputs[tp][k]);
				}

			}//end for

			//calculate error and return as percentage
			return mse / (NN.nOutput * tset.size);
		}
		public unsafe double getRandomMSE(int size)
		{
			double mse = 0;

			//for every training input array
			for(int tp = 0; tp < size; tp++)
			{
				int index = rand.Next(tset.size);
				//feed inputs through network and backpropagate errors
				NN.feedForward(tset.inputs[index]);

				//check all outputs against desired output values
				for(int k = 0; k < NN.nOutput; k++)
				{
					//sum all the MSEs together
					mse += Abs(NN.outputNeurons[k] - tset.outputs[index][k]);
				}

			}//end for

			//calculate error and return as percentage
			return mse / (NN.nOutput * size);
		}
		public unsafe double getAccuracy(double level)
		{
			double incorrectResults = 0;
			int size = tset.size;

			//for every training input array
			for(int tp = 0; tp < size; tp++)
			{
				//feed inputs through network and backpropagate errors
				NN.feedForward(tset.inputs[tp]);

				//correct pattern flag

				//check all outputs against desired output values
				for(int k = 0; k < NN.nOutput; k++)
				{
					//set flag to false if desired and output differ
					if(NN.outputNeurons[k] > tset.outputs[tp][k] + level || NN.outputNeurons[k] < tset.outputs[tp][k] - level)
					{
						incorrectResults++;
						break;
					}
				}
				//inc training error for a incorrect result
			}//end for

			//calculate error and return as percentage
			return 1 - incorrectResults / (double)(size - 1);
		}
		public unsafe double getRandomAccuracy(double level, int size)
		{
			int incorrectResults = 0;

			//for every training input array
			for(int tp = 0; tp < size; tp++)
			{
				int index = rand.Next(tset.size);
				//feed inputs through network and backpropagate errors
				NN.feedForward(tset.inputs[index]);

				//correct pattern flag

				//check all outputs against desired output values
				for(int k = 0; k < NN.nOutput; k++)
				{
					//set flag to false if desired and output differ
					if(NN.outputNeurons[k] > tset.outputs[index][k] + level || NN.outputNeurons[k] < tset.outputs[index][k] - level)
					{
						incorrectResults++;
						break;
					}
				}
				//inc training error for a incorrect result
			}//end for

			//calculate error and return as percentage
			return 1 - incorrectResults / (double)(size - 1);
		}

		protected static readonly System.Random rand = new System.Random();
	}
	public class trainerModern
	{
		public readonly vnn NN;
		protected double[] outputErrorGradients, hiddenErrorGradients;
		protected double[,] deltaHiddenOutput, deltaInputHidden;
		public trainerModern(vnn net)
		{
			NN = net;

			outputErrorGradients = new double[net.nOutput];
			hiddenErrorGradients = new double[net.nHidden];

			deltaHiddenOutput = new double[net.NOutput, net.nHidden + 1];
            deltaInputHidden = new double[net.NHidden, net.nInput + 1];
		}

		public virtual void backpropagate(double[] desiredOutputs, double learningRate, double momentum)
		{
			//modify deltas between hidden and output layers
			//--------------------------------------------------------------------------------------------------------
			for (int k = 0; k < NN.nOutput; k++)
			{
				//get error gradient for every output node
				outputErrorGradients[k] = NN.outputNeurons[k] * (1 - NN.outputNeurons[k]) * (desiredOutputs[k] - NN.outputNeurons[k]);
				//for all nodes in hidden layer and bias neuron
				for (int j = 0; j <= NN.nHidden; j++)
				{
					//calculate change in weight
					deltaHiddenOutput[k, j] = learningRate * NN.hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[k, j];
					//vmon.Line("[H] neuron value = " + NN.hiddenNeurons[j] + "; neuron delta = " + deltaHiddenOutput[j, k] + "; error gradient = " + outputErrorGradients[k]);
				}
			}

			//modify deltas between input and hidden layers
			//--------------------------------------------------------------------------------------------------------
			for (int j = 0; j < NN.nHidden; j++)
			{
				//get error gradient for every hidden node
				double weightedSum = 0;
				for (int k = 0; k < NN.nOutput; k++) weightedSum += NN.wHiddenOutput[k, j] * outputErrorGradients[k];
				hiddenErrorGradients[j] = NN.hiddenNeurons[j] * (1 - NN.hiddenNeurons[j]) * weightedSum;

				//for all nodes in input layer and bias neuron
				for (int i = 0; i <= NN.nInput; i++)
				{
                    //calculate change in weight 
                    deltaInputHidden[j, i] = learningRate * NN.inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[j, i];
					//vmon.Line("[I] neuron value = " + NN.inputNeurons[i] + "; neuron delta = " + deltaInputHidden[i, j] + "; error gradient = " + hiddenErrorGradients[j]);
				}
			}
			//vmon.Line("");

			//if using stochastic learning update the weights immediately
			updateWeights();
		}
		protected virtual void updateWeights()
		{
			//input -> hidden weights
			//--------------------------------------------------------------------------------------------------------
			for (int i = 0; i <= NN.nInput; i++)
			{
				for (int j = 0; j < NN.nHidden; j++)
				{
					//update weight
					NN.wInputHidden[j, i] += deltaInputHidden[j, i];
				}
			}

			//hidden -> output weights
			//--------------------------------------------------------------------------------------------------------
			for (int j = 0; j <= NN.nHidden; j++)
			{
				for (int k = 0; k < NN.nOutput; k++)
				{
					//update weight
					NN.wHiddenOutput[k, j] += deltaHiddenOutput[k, j];
				}
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void TrainOne(double[] inputs, double[] desiredOutputs, double learningRate, double momentum)
		{
			NN.feedForward(inputs);
			backpropagate(desiredOutputs, learningRate, momentum);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void TrainEpoch(double[][] inputs, double[][] desiredOutputs, double learingRate, double momentum)
		{
			for(int i = 0, n = inputs.Length; i < n; i++)
			{
				TrainOne(inputs[i], desiredOutputs[i], learingRate, momentum);
			}
		}

		protected static readonly System.Random rand = new System.Random();
	}
    public class trainerNoMomentum : ITrainer
    {
        public readonly vnn NN;
        readonly double[] outputErrorGradients, hiddenErrorGradients;
        public trainerNoMomentum(vnn nn)
        {
            NN = nn;
            outputErrorGradients = new double[NN.nOutput];
            hiddenErrorGradients = new double[NN.nHidden];
        }
		
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
    	public void backpropagate(double[] desiredOutputs, double learningRate)
        {
            backpropagate(NN, outputErrorGradients, hiddenErrorGradients, desiredOutputs, learningRate);
            //backpropagate_old(desiredOutputs, learningRate);
        }
     	public unsafe static void backpropagate(vnn NN, double[] outputErrorGradients, double[] hiddenErrorGradients, double[] desiredOutputs, double learningRate)
        {
            getOEG(NN.outputNeurons, desiredOutputs, outputErrorGradients);
            mult(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, learningRate, NN.nHidden, NN.nOutput);

            getHEG(NN.wHiddenOutput, NN.hiddenNeurons, outputErrorGradients, hiddenErrorGradients, NN.NOutput, NN.nHidden);
            mult(NN.wInputHidden, NN.inputNeurons, hiddenErrorGradients, learningRate, NN.nInput, NN.NHidden);
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
     	public static unsafe void getOEG(double[] output, double[] desired, double[] gradient)
        {
            double* outpos, OEGpos, despos;
            fixed (double* _outpos = output) { outpos = _outpos; }
            fixed (double* _OEGpos = gradient) { OEGpos = _OEGpos; }
            fixed (double* _despos = desired) { despos = _despos; }

            for (int k = 0, to = output.Length; k < to; k++)
            {
                double o = (*outpos++);
                (*OEGpos++) = o * (1 - o) * ((*despos++) - o);
            }
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
     	public static unsafe void getHEG(double[,] wHiddenOutput, double[] hiddenNeurons, double[] outputErrorGradients, double[] hiddenErrorGradients, int nOutput, int nHidden)
        {
            double weightedSum;
            double* PREVGRADp, GRADp, Wp, NEURONSp;
            fixed (double* _Wp = &wHiddenOutput[0, 0]) { Wp = _Wp; }
            fixed (double* _Neuronsp = &hiddenNeurons[0]) { NEURONSp = _Neuronsp; }
            fixed (double* _GRADp = hiddenErrorGradients) { GRADp = _GRADp; }

            for (int j = 0; j < nHidden; j++, NEURONSp++)
            {
                fixed (double* _PREVGRADp = outputErrorGradients) { PREVGRADp = _PREVGRADp; }
                weightedSum = 0;
                for (int k = 0; k < nOutput; k++)
                {
                    //weightedSum += wHiddenOutput[k, j] * (*OEGpos++);
                    weightedSum += (Wp[k * nOutput + j]) * (*PREVGRADp++);
                }
                (*GRADp++) = (*NEURONSp) * (1 - (*NEURONSp)) * weightedSum;
            }
        }
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
     	public static unsafe void mult(double[,] W, double[] neurons, double[] gradient, double learningRate, int insize, int outsize)
        {
            double* npos, npos0, Wpos, OEGpos;
            fixed (double* _Wpos = &W[0, 0]) { Wpos = _Wpos; }
            fixed (double* _gpos = gradient) { OEGpos = _gpos; }
			fixed (double* npos_ = &neurons[0]) { npos0 = npos_; }

            for (int k = 0; k < outsize; k++)
            {
                double lr_times_grad = (*OEGpos++) * learningRate;
				npos = npos0;

                for (int j = 0; j <= insize; j++)
                {
                    (*Wpos++) += (*npos++) * lr_times_grad;
                    //NN.wHiddenOutput[k, j] += learningRate * NN.hiddenNeurons[j] * outputErrorGradients[k];
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void TrainOne(double[] inputs, double[] desiredOutputs, double learningRate)
        {
            NN.feedForward(inputs);
            backpropagate(desiredOutputs, learningRate);
        }
    }
}
