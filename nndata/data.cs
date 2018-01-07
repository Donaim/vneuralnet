using static System.Math;
using VNNLib;
using System;
using System.Linq;
namespace VNNAddOn
{
	public static class data
	{
		public static readonly Random rand = new Random();
		public static class ActivationFunctions
		{
			public static Func<double, double> Sigmoid = (x) => 1 / (1 + Exp(-x));
			public static Func<double, double> Tan = (x) => Tanh(x);
			public static Func<double, double> Step = (x) => x > 0.5 ? 1 : 0;
		}
		public static class RandomizingFunctions
		{
			public static void SigmoidFull(vnn nn)
			{
				for(int i = 0; i <= nn.nInput; i++)
				{
					for(int j = 0; j < nn.nHidden; j++)
					{
						nn.wInputHidden[i, j] = (rand.NextDouble() * 12 - 6) / (nn.nInput + 1);
					}
				}
				for(int i = 0; i <= nn.nHidden; i++)
				{
					for(int j = 0; j < nn.nOutput; j++)
					{
						nn.wHiddenOutput[i, j] = (rand.NextDouble() * 12 - 6) / (nn.nHidden + 1);
					}
				}
			}
			public static void StepFull(vnn nn)
			{
				for(int i = 0; i <= nn.nInput; i++)
				{
					for(int j = 0; j < nn.nHidden; j++)
					{
						nn.wInputHidden[i, j] = rand.NextDouble() / nn.nInput;
					}
				}
				for(int i = 0; i <= nn.nHidden; i++)
				{
					for(int j = 0; j < nn.nOutput; j++)
					{
						nn.wHiddenOutput[i, j] = rand.NextDouble() / nn.nHidden;
					}
				}
			}
		}
		public static class DataSets
		{
			public static trainingSet xorProblem
			{
				get
				{
					double[][] inp;
					double[][] outp;

					inp = new double[4][];
					inp[0] = new double[] { 0, 0 };
					inp[1] = new double[] { 0, 1 };
					inp[2] = new double[] { 1, 0 };
					inp[3] = new double[] { 1, 1 };

					outp = new double[4][];
					outp[0] = new double[] { 0 };
					outp[1] = new double[] { 1 };
					outp[2] = new double[] { 1 };
					outp[3] = new double[] { 0 };

					return new trainingSet(inp, outp);
				}
			}
			public static trainingSet sinxcFunction
			{
				get
				{
					double[][] inp = new double[1000][], outp = new double[1000][];
					for(int i = 0; i < 1000; i++)
					{
						double x; do { x = rand.NextDouble(); } while(x == 0);
						inp[i] = new double[] { x };
						outp[i] = new double[] { Sin(x) / x };
					}

					return new trainingSet(inp, outp);
				}
			}
			public static trainingSet twoParamTest
			{
				get
				{
					var rand = new Random();
					int n = 1000;
					double[][] inp = new double[n][];
					double[][] outp = new double[n][];

					for(int i = 0; i < n; i++)
					{
						double d1 = rand.NextDouble(), d2 = rand.NextDouble();
						inp[i] = new double[] { d1, d2 };
						outp[i] = new double[] { d1 > 0.5 ? 1 : 0, Sin(d1 + d2) * d1 };
					}

					return new trainingSet(inp, outp);
				}
			}
			public static trainingSet primeTest
			{
				get
				{
					var rand = new Random();
					int n = 10000;
					double[][] inp = new double[n][];
					double[][] outp = new double[n][];

					for(int i = 0; i < n; i++)
					{
						double d = rand.NextDouble();
						inp[i] = new double[] { d };
						outp[i] = new double[] { int.Parse(d.ToString().PadRight(20, '0')[5].ToString()) > 5 ? 1 : 0 };
					}

					return new trainingSet(inp, outp);
				}
			}
			public static trainingSet sorting(int n)
			{
				int l = 1000;

				double[][] inp = new double[l][], outp = new double[l][];
				for(int i = 0; i < l; i++)
				{
					outp[i] = new double[n];
					inp[i] = new double[n];
					double[] temp = new double[n];

					for(int j = 0; j < n; j++)
					{
						temp[j] = rand.NextDouble();
					}
					inp[i] = temp.Select(z => z).ToArray();
					var ordered = temp.OrderBy(z => z).ToList();
					for(int j = 0; j < n; j++)
					{
						outp[i][j] = ordered.IndexOf(inp[i][j]) / (double)(n - 1);
					}

					//Console.WriteLine('{' + string.Join(", ", inp[i].Select(z => z.ToString("N2"))) + '}');
					//Console.WriteLine('{' + string.Join(", ", outp[i].Select(z => z.ToString("N2"))) + '}');
					//Console.WriteLine();
				}

				return new trainingSet(inp, outp);
			}
			public static trainingSet encryption
			{
				get
				{
					double[][] inp = new double[1000][], outp = new double[1000][];
					for(int i = 0; i < 1000; i++)
					{
						double x; do { x = rand.NextDouble(); } while(x == 0);
						inp[i] = new double[] { x };
						outp[i] = new double[] { Sin(x) / x };
					}

					return new trainingSet(inp, outp);
				}
			}
			struct sortinghelp
			{
				public readonly int index;
				public readonly double wage;
				public sortinghelp(int i, double w) { index = i; wage = w; }
			}
		}
		public static class CustomClasses
		{
			public class randomUpdateTrainer : trainer
			{
				public randomUpdateTrainer(vnn nn, trainingSet tset, double learningRate, double momentum) : base(nn, tset, learningRate, momentum) { }
				protected override void updateWeights()
				{
					for(int i = 0; i <= NN.nInput; i++)
					{
						for(int j = 0; j < NN.nHidden; j++)
						{
							if(Abs(NN.wInputHidden[i, j]) < 0.5 / NN.nInput) { NN.wInputHidden[i, j] = (rand.NextDouble() * 12 - 6) / NN.nInput; }
							else
							{ NN.wInputHidden[i, j] += deltaInputHidden[i, j]; }
						}
					}
					for(int j = 0; j <= NN.nHidden; j++)
					{
						for(int k = 0; k < NN.nOutput; k++)
						{
							if(Abs(NN.wHiddenOutput[j, k]) < 0.5 / NN.nHidden) { NN.wHiddenOutput[j, k] = (rand.NextDouble() * 12 - 6) / NN.nInput; }
							else
							{ NN.wHiddenOutput[j, k] += deltaHiddenOutput[j, k]; }
						}
					}
				}
			}
			public static class DynamicSynapsis
			{
				public class DynamicVNN : vnn
				{
					public int aInputHidden = 0, aHiddenOutput = 0;
					public double liveRatio;
					public bool[,] bInputHidden, bHiddenOutput;
					public DynamicVNN(int ni, int nh, int no, double lRatio) : base(ni, nh, no)
					{
						bInputHidden = new bool[ni + 1, nh];
						bHiddenOutput = new bool[nh + 1, no];

						liveRatio = lRatio;
						randomize_dynamic();
					}
					void randomize_dynamic()
					{
						for(int i = 0; i <= nInput; i++)
						{
							for(int j = 0; j < nHidden; j++)
							{
								if(rand.NextDouble() < liveRatio)
								{
									bInputHidden[i, j] = true;
									wInputHidden[i, j] = (rand.NextDouble() * 12 - 6) / (nInput + 1);
									aInputHidden++;
								}
							}
						}
						for(int i = 0; i <= nHidden; i++)
						{
							for(int j = 0; j < nOutput; j++)
							{
								if(rand.NextDouble() < liveRatio)
								{
									bHiddenOutput[i, j] = true;
									wHiddenOutput[i, j] = (rand.NextDouble() * 12 - 6) / (nHidden + 2);
									aHiddenOutput++;
								}
							}
						}
					}
					public override unsafe void feedForward(double[] pattern)
					{
						for(int i = 0; i < nInput; i++)
						{
							inputNeurons[i] = pattern[i];
							//vmon.Line("feeded = " + inputNeurons[i] + "; pattern = " + pattern[i]);
						}

						//Calculate Hidden Layer values - include bias neuron
						//--------------------------------------------------------------------------------------------------------
						for(int j = 0; j < nHidden; j++)
						{
							//clear value
							hiddenNeurons[j] = 1.0 / nHidden;

							//get weighted sum of pattern and bias neuron
							for(int i = 0; i <= nInput; i++)
							{
								if(bInputHidden[i, j])
								{
									hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i, j];
								}
							}

							//set to result of sigmoid
							hiddenNeurons[j] = 1 / (1 + Exp(-hiddenNeurons[j])); //activation function
						}

						//Calculating Output Layer values - include bias neuron
						//--------------------------------------------------------------------------------------------------------
						for(int k = 0; k < nOutput; k++)
						{
							//clear value
							outputNeurons[k] = 0;

							//get weighted sum of pattern and bias neuron
							for(int j = 0; j <= nHidden; j++)
							{
								if(bHiddenOutput[j, k])
								{
									outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j, k];
								}
							}
							//set to result of sigmoid
							outputNeurons[k] =  1 / (1 + Exp(-outputNeurons[k])); //activation function
						}
					}
				}
				public class DynamicTrainer : trainer
				{
					DynamicVNN dnn;
					public DynamicTrainer(DynamicVNN nn, trainingSet tset, double learningrate, double moment) : base(nn, tset, learningrate, moment)
					{
						dnn = nn;
					}

					public void TrainDynamicallyFor(int n)
					{
						var updatingTask = new System.Threading.Thread(updateDynamics);
						updatingTask.Start();
						for(int i = 0, l = tset.size; i < n; i++)
						{
							NN.feedForward(tset.inputs[i % l]);
							backpropagate(tset.outputs[i % l]);
						}
						updatingTask.Abort();
					}
					void updateDynamics()
					{
						while(true)
						{
							for(int i = 0; i <= NN.nInput; i++)
							{
								for(int j = 0; j < NN.nHidden; j++)
								{
									if(dnn.bInputHidden[i, j])
									{
										if(Abs(NN.wInputHidden[i, j]) < 0.1 / (NN.nInput + 1)) { dnn.bInputHidden[i, j] = false; dnn.aInputHidden--; }
									}
								}
							}
							for(int j = 0; j <= NN.nHidden; j++)
							{
								for(int k = 0; k < NN.nOutput; k++)
								{
									if(dnn.bHiddenOutput[j, k])
									{
										if(Abs(NN.wHiddenOutput[j, k]) < 0.1 / (NN.nHidden + 1)) { dnn.bHiddenOutput[j, k] = false; dnn.aHiddenOutput--; }
									}
								}
							}

							//if(dnn.aInputHidden / (double)((NN.nInput + 1) * NN.nHidden) < dnn.liveRatio)
							{
								int id1, id2;
								do
								{
									id1 = rand.Next(NN.nInput + 1);
									id2 = rand.Next(NN.nHidden);
								}
								while(dnn.bInputHidden[id1, id2]);

								NN.wInputHidden[id1, id2] = (rand.NextDouble() * 12 - 6) / NN.nInput;
								dnn.bInputHidden[id1, id2] = true;
								dnn.aInputHidden++;
								deltaInputHidden[id1, id2] = 0;
							}
							//if(dnn.aHiddenOutput / (double)((NN.nHidden + 1) * NN.nOutput) < dnn.liveRatio)
							{
								int id1, id2;
								do
								{
									id1 = rand.Next(NN.nHidden + 1);
									id2 = rand.Next(NN.nOutput);
								}
								while(dnn.bHiddenOutput[id1, id2]);

								NN.wHiddenOutput[id1, id2] = (rand.NextDouble() * 12 - 6) / NN.nHidden;
								dnn.bHiddenOutput[id1, id2] = true;
								dnn.aHiddenOutput++;
								deltaHiddenOutput[id1, id2] = 0;
							}
						}
					}

					public override unsafe void backpropagate(double[] desiredOutputs)
					{
						for(int k = 0; k < NN.nOutput; k++)
						{
							outputErrorGradients[k] = NN.outputNeurons[k] * (1 - NN.outputNeurons[k]) * (desiredOutputs[k] - NN.outputNeurons[k]);
							for(int j = 0; j <= NN.nHidden; j++)
							{
								if(dnn.bHiddenOutput[j, k])
								{
									deltaHiddenOutput[j, k] = learningRate * NN.hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j, k];
								}
							}
						}
						for(int j = 0; j < NN.nHidden; j++)
						{
							double weightedSum = 0;
							for(int k = 0; k < NN.nOutput; k++) weightedSum += NN.wHiddenOutput[j, k] * outputErrorGradients[k];
							hiddenErrorGradients[j] = NN.hiddenNeurons[j] * (1 - NN.hiddenNeurons[j]) * weightedSum;
							for(int i = 0; i <= NN.nInput; i++)
							{
								if(dnn.bInputHidden[i, j])
								{
									deltaInputHidden[i, j] = learningRate * NN.inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i, j];
								}
							}
						}
						updateWeights();
					}
					protected override void updateWeights()
					{
						for(int i = 0; i <= NN.nInput; i++)
						{
							for(int j = 0; j < NN.nHidden; j++)
							{
								if(dnn.bInputHidden[i, j]) { NN.wInputHidden[i, j] += deltaInputHidden[i, j]; }
							}
						}
						for(int j = 0; j <= NN.nHidden; j++)
						{
							for(int k = 0; k < NN.nOutput; k++)
							{
								if(dnn.bHiddenOutput[j, k]) { NN.wHiddenOutput[j, k] += deltaHiddenOutput[j, k]; }
							}
						}
					}
				}
			}
		}
	}
}
