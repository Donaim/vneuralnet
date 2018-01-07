using VNNLib;
using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;
using System.IO;

namespace VNNAddOn
{
	public static class addon
	{
		public static void LoadWeights(this vnn t, string str)
		{
			using(var reader = new StreamReader(str, System.Text.Encoding.ASCII))
			{
				for(int i = 0; i <= t.NInput; i++)
				{
					for(int j = 0; j < t.NHidden; j++)
					{
						//set weights to random values
						t.GwInputHidden[j, i] = double.Parse(reader.ReadLine());
					}
				}

				//set weights between input and hidden
				//--------------------------------------------------------------------------------------------------------
				for(int i = 0; i <= t.NHidden; i++)
				{
					for(int j = 0; j < t.NOutput; j++)
					{
						//set weights to random values
						t.GwHiddenOutput[j, i] = double.Parse(reader.ReadLine());
					}
				}
			}
		}
        public static void SaveWeights(this vnn t, string str)
		{
			using(var writer = new System.IO.StreamWriter(str, false, System.Text.Encoding.ASCII))
			{
				for(int i = 0; i <= t.NInput; i++)
				{
					for(int j = 0; j < t.NHidden; j++)
					{
						//set weights to random values
						writer.WriteLine(t.GwInputHidden[j, i]);
					}
				}

				//set weights between input and hidden
				//--------------------------------------------------------------------------------------------------------
				for(int i = 0; i <= t.NHidden; i++)
				{
					for(int j = 0; j < t.NOutput; j++)
					{
						//set weights to random values
						writer.WriteLine(t.GwHiddenOutput[j, i]);
					}
				}
			}
		}

        public static void RandomizeWeights(ISimpleMLP nn)
		{
			Random rand = new Random();
			//set weights between input and hidden 		
			//--------------------------------------------------------------------------------------------------------
			for(int i = 0; i <= nn.NInput; i++)
			{
				for(int j = 0; j < nn.NHidden; j++)
				{
					//set weights to random values
					nn.GwInputHidden[j, i] = (rand.NextDouble() * 12 - 6) / nn.NInput;
				}
			}

			//set weights between input and hidden
			//--------------------------------------------------------------------------------------------------------
			for(int i = 0; i <= nn.NHidden; i++)
			{
				for(int j = 0; j < nn.NOutput; j++)
				{
					//set weights to random values
					nn.GwHiddenOutput[j, i] = (rand.NextDouble() * 12 - 6) / nn.NHidden;
				}
			}
		}

        /// <summary>
        /// Greather mults means greather variance from 0 after activation function
        /// </summary>
        /// <param name="mult1">Optimals are: 1.5 for inputs close to 1 and 5.0 for 10% ones and rest zeroes</param>
        /// <param name="mult2">Optimal is 3.5 for all sets of inputs if prev layers was good</param>
        public static void RandomizeUniform(ISimpleMLP nn, double mult1 = 5, double mult2 = 3.5)
        {
            Random rand = new Random();
            //set weights between input and hidden 		
            //--------------------------------------------------------------------------------------------------------
            double limit = Sqrt(3.0 / nn.NInput) * mult1;
            for (int i = 0; i <= nn.NInput; i++)
            {
                for (int j = 0; j < nn.NHidden; j++)
                {
                    //set weights to random values
                    nn.GwInputHidden[j, i] = (rand.NextDouble() * 2 - 1) * limit;
                }
            }

            //set weights between input and hidden
            //--------------------------------------------------------------------------------------------------------
            limit = Sqrt(3.0 / nn.NHidden) * mult2;
            for (int i = 0; i <= nn.NHidden; i++)
            {
                for (int j = 0; j < nn.NOutput; j++)
                {
                    //set weights to random values
                    nn.GwHiddenOutput[j, i] = (rand.NextDouble() * 2 - 1) * limit;
                }
            }
        }

        public static void TestLoop(this vnn nn)
		{
			try
			{
				while(true)
				{
					double[] inp = new double[nn.NInput];
					for(int i = 0; i < nn.NInput; i++)
					{
						System.Console.Write("in[" + i + "] = ");
						inp[i] = double.Parse(System.Console.ReadLine());
					}
					System.Console.WriteLine();
					double[] ans = nn.feedResult(inp);
					for(int i = 0; i < nn.NOutput; i++)
					{
						Console.WriteLine("ou[" + i + "] = " + Round(ans[i], 5));
					}
					Console.WriteLine("".PadRight(10, '='));
				}
			} catch { }
		}
		public static void TestLoop(this vnn nn, Func<double[], double[]> answerkey)
		{
			try
			{
				while(true)
				{
					double[] inp = new double[nn.NInput];
					for(int i = 0; i < nn.NInput; i++)
					{
						Console.Write("in[" + i + "] = ");
						inp[i] = double.Parse(System.Console.ReadLine());
					}
					System.Console.WriteLine();
					double[] ans = nn.feedResult(inp);
					double[] corr = answerkey(ans);
					for(int i = 0; i < nn.NOutput; i++)
					{
						Console.WriteLine("ou[{0}] = {1:0.000}; co[{0}] = {2:0.000}; dx[{0}] = {3:0.000};", i, ans[i], corr[i], Abs(corr[i] - ans[i]));
					}
					Console.WriteLine("".PadRight(10, '='));
				}
			} catch { }
		}
		public static void TestLoop(this vnn nn, Func<double[], double[]> answerkey, Func<string, double[]> translator)
		{
			try
			{
				while(true)
				{
					double[] inp = new double[nn.NInput];

					Console.Write("in = ");
					inp = translator(Console.ReadLine());
					Console.WriteLine();

					double[] ans = nn.feedResult(inp);
					double[] corr = answerkey(ans);
					for(int i = 0; i < nn.NOutput; i++)
					{
						Console.WriteLine("ou[{0}] = {1:0.000}; co[{0}] = {2:0.000}; dx[{0}] = {3:0.000};", i, ans[i], corr[i], Abs(corr[i] - ans[i]));
					}
					Console.WriteLine("".PadRight(10, '='));
				}
			} catch { }
		}
		public static void TestLoop(this vnn nn, Func<string, double[]> input_translator, Func<double[], string> output_translator)
		{
			try
			{
				while(true)
				{
					double[] inp = new double[nn.NInput];

					Console.Write("in = ");
					inp = input_translator(Console.ReadLine());
					Console.WriteLine();

					double[] ans = nn.feedResult(inp);
					Console.WriteLine("out = " + output_translator(ans));
					Console.WriteLine("".PadRight(10, '='));
				}
			} catch { }
		}
	}
	public class reporter
	{
		public vnn who;
		public reporter(vnn on_who) { who = on_who; }

		public string wHiddenOutput
		{
			get
			{
				string re = string.Empty;

				for(int i = 0; i <= who.NHidden; i++)
				{
					for(int j = 0; j < who.NOutput; j++)
					{
						re += who.GwHiddenOutput[i, j].ToString("N3").PadLeft(7);
					}
					re += '\n';
				}

				return re;
			}
		}
		public string wInputHidden
		{
			get
			{
				string re = string.Empty;

				for(int i = 0; i <= who.NInput; i++)
				{
					for(int j = 0; j < who.NHidden; j++)
					{
						re += who.GwInputHidden[i, j].ToString("N3").PadLeft(7);
					}
					re += '\n';
				}

				return re;
			}
		}

		public unsafe string inputNeurons
		{
			get
			{
				string re = string.Empty;

				for(int i = 0; i < who.NInput; i++)
				{
					re += who.GInputNeurons[i] + "\n";
				}

				return re;
			}
		}
		public unsafe string hiddenNeurons
		{
			get
			{
				string re = string.Empty;

				for(int i = 0; i < who.NHidden; i++)
				{
					re += who.GHiddenNeurons[i] + "\n";
				}

				return re;
			}
		}
		public unsafe string outputNeurons
		{
			get
			{
				string re = string.Empty;

				for(int i = 0; i < who.NOutput; i++)
				{
					re += who.GOutputNeurons[i] + "\n";
				}

				return re;
			}
		}
	}
}
