using System.Runtime.CompilerServices;

using static System.Math;
using VNNLib;

namespace VNNAddOn {
    public struct trainingSet
	{
		public readonly int size;
		public readonly double[][] inputs;
		public readonly double[][] outputs;
		public trainingSet(double[][] inp, double[][] outp)
		{
			inputs = inp;
			outputs = outp;
			if(inp.Length != outputs.Length) { throw new exceptions.TSetDifferentInputs(); }
			size = inp.Length;
		}
		public static trainingSet fromFile(string csv_file, int inputs_n)
		{
			string[] lines = System.IO.File.ReadAllLines(csv_file, System.Text.Encoding.ASCII);
			double[][] inp = new double[lines.Length][], outp = new double[lines.Length][];
			for(int i = 0; i < lines.Length; i++)
			{
				inp[i] = new double[inputs_n];
				string[] splitted = lines[i].Split(',');
				for(int j = 0; j < inputs_n; j++)
				{
					inp[i][j] = double.Parse(splitted[j]);
				}
				outp[i] = new double[splitted.Length - inputs_n];
				for(int j = inputs_n; j < splitted.Length; j++)
				{
					outp[i][j - inputs_n] = double.Parse(splitted[j]);
				}
			}

			return new trainingSet(inp, outp);
		}

		private static System.Random rand = new System.Random();
		public void Shuffle()
		{
			for(int i = size - 1; i > 1; i--)
			{
				int j = rand.Next(i + 1);
				var ii = inputs[i];
				var oi = outputs[i];

				inputs[i] = inputs[j];
				outputs[i] = outputs[j];

				inputs[j] = ii;
				outputs[j] = oi;
			}
		}

		public unsafe double getMSE(vnn NN)
		{
			double mse = 0;

			//for every training input array
			for (int tp = 0, l = size; tp < l; tp++)
			{
				//feed inputs through network and backpropagate errors
				NN.feedForward(inputs[tp]);

				//check all outputs against desired output values
				for (int k = 0; k < NN.nOutput; k++)
				{
					//sum all the MSEs together
					mse += Abs(NN.outputNeurons[k] - outputs[tp][k]);
				}

			}//end for

			//calculate error and return as percentage
			return mse / (NN.nOutput * size);
		}
		public unsafe double getRandomMSE(vnn NN, int sapmleSize)
		{
			double mse = 0;

			//for every training input array
			for (int tp = 0; tp < sapmleSize; tp++)
			{
				int index = rand.Next(size);
				//feed inputs through network and backpropagate errors
				NN.feedForward(inputs[index]);

				//check all outputs against desired output values
				for (int k = 0; k < NN.nOutput; k++)
				{
					//sum all the MSEs together
					mse += Abs(NN.outputNeurons[k] - outputs[index][k]);
				}

			}//end for

			//calculate error and return as percentage
			return mse / (NN.nOutput * sapmleSize);
		}
		public unsafe double getAccuracy(vnn NN, double level)
		{
			double incorrectResults = 0;

			//for every training input array
			for (int tp = 0; tp < size; tp++)
			{
				//feed inputs through network and backpropagate errors
				NN.feedForward(inputs[tp]);

				//correct pattern flag

				//check all outputs against desired output values
				for (int k = 0; k < NN.nOutput; k++)
				{
					//set flag to false if desired and output differ
					if (NN.outputNeurons[k] > outputs[tp][k] + level || NN.outputNeurons[k] < outputs[tp][k] - level)
					{
						incorrectResults++;
						break;
					}
				}
				//inc training error for a incorrect result
			}//end for

			//calculate error and return as percentage
			return 1 - incorrectResults / (double)(size);
		}
		public unsafe double getRandomAccuracy(vnn NN, double level, int sampleSize)
		{
			int incorrectResults = 0;

			//for every training input array
			for (int tp = 0; tp < sampleSize; tp++)
			{
				int index = rand.Next(size);
				//feed inputs through network and backpropagate errors
				NN.feedForward(inputs[index]);

				//correct pattern flag

				//check all outputs against desired output values
				for (int k = 0; k < NN.nOutput; k++)
				{
					//set flag to false if desired and output differ
					if (NN.outputNeurons[k] > outputs[index][k] + level || NN.outputNeurons[k] < outputs[index][k] - level)
					{
						incorrectResults++;
						break;
					}
				}
				//inc training error for a incorrect result
			}//end for

			//calculate error and return as percentage
			return 1 - incorrectResults / (double)(sampleSize);
		}
	}
}