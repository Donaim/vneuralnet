using static System.Math;

using System;

namespace VNNLib
{
    public class vnn : IFeedResultNN, IFeedForwardNN, ISimpleMLP, ICopyableNN<vnn>
	{
        public vnn(double[,] wIH, double[,] wHO, Action<ISimpleMLP> randFunc)
		{
			nInput = wIH.GetLength(1) - 1;
            nHidden = wIH.GetLength(0); if(wIH.GetLength(0) != wHO.GetLength(1) - 1) { throw new System.Exception("Incorrect arrays dimensions"); }
			nOutput = wHO.GetLength(0);

			inputNeurons = new double[nInput + 1];
			inputNeurons[nInput] = 1;

			hiddenNeurons = new double[nHidden + 1];
			hiddenNeurons[nHidden] = 1;

			outputNeurons = new double[nOutput];

			wInputHidden = wIH;
			wHiddenOutput = wHO;

            randFunc(this);
        }

		public readonly int nInput, nHidden, nOutput;
		public readonly double[] inputNeurons, hiddenNeurons, outputNeurons;
		public readonly unsafe double[,] wInputHidden, wHiddenOutput;

        public double[,] GwInputHidden => wInputHidden;
        public double[,] GwHiddenOutput => wHiddenOutput;
        public double[] GInputNeurons => inputNeurons;
        public double[] GHiddenNeurons => hiddenNeurons;
        public double[] GOutputNeurons => outputNeurons;
        public int NInput => nInput;
        public int NHidden => nHidden;
        public int NOutput => nOutput;

        public vnn(int nI, int nH, int nO) : this(new double[nH, nI + 1], new double[nO, nH + 1], (nn) => { }) { }
		public vnn(int nI, int nH, int nO, Action<ISimpleMLP> randFunc) : this(new double[nH, nI + 1], new double[nO, nH + 1], randFunc) { }
		public vnn(double[,] wIH, double[,] wHO) : this(wIH, wHO, (nn) => { }) { }

		public virtual unsafe void feedForward(double[] pattern)
		{
            //set input neurons to input values
            Array.Copy(pattern, 0, inputNeurons, 0, nInput);
            
            mult(wInputHidden, inputNeurons, hiddenNeurons);
            mult(wHiddenOutput, hiddenNeurons, outputNeurons);
        }
        public static unsafe void mult(double[,] A, double[] x, double[] y)
        {
            double ytemp;
            double* xpos, xpos0, Apos, ypos;
            fixed (double* Apos_ = &A[0, 0]) { Apos = Apos_; }
            fixed (double* ypos_ = &y[0]) { ypos = ypos_; }
			fixed (double* xpos_ = &x[0]) { xpos0 = xpos_; }

			int insize = x.Length;
			int outsize = y.Length;

            for (int i = 0; i < outsize; ++i)
            {
				xpos = xpos0;
                ytemp = 0;

                for (int j = 0; j < insize; ++j)
                {
                    ytemp += (*Apos++) * (*xpos++);
                }

                (*ypos++) = 1.0 / (1.0 + Exp(-ytemp));
            }
        }

        public double[] feedResult(double[] pattern)
		{
			feedForward(pattern);

			//create copy of output results
			double[] results = new double[nOutput];
            Array.Copy(outputNeurons, 0, results, 0, nOutput);

			return results;
		}

        public unsafe byte[] ToBytes()
		{
			byte[] re = new byte[
				+ sizeof(int) * 3 //sizes
				+ sizeof(double) * (nInput + 1) * nHidden + sizeof(double) * (nHidden + 1) * nOutput //weights
				];

			using(var stream = new System.IO.MemoryStream(re)) {
			using(var writer = new System.IO.BinaryWriter(stream))
			{
				writer.Write(nInput);
				writer.Write(nHidden);
				writer.Write(nOutput);

				for(int i = 0; i <= nInput; i++)
				{
					for(int j = 0; j < nHidden; j++)
					{
						writer.Write(wInputHidden[j, i]);
					}
				}
				for(int i = 0; i <= nHidden; i++)
				{
					for(int j = 0; j < nOutput; j++)
					{
                        writer.Write(wHiddenOutput[j, i]);
					}
				}
			}}

			return re;
		}
		public vnn(byte[] raw)
		{
			using(var stream = new System.IO.MemoryStream(raw)) {
			using(var reader = new System.IO.BinaryReader(stream))
			{
				nInput = reader.ReadInt32();
				nHidden = reader.ReadInt32();
				nOutput = reader.ReadInt32();

				inputNeurons = new double[nInput + 1];
				inputNeurons[nInput] = 1;

				hiddenNeurons = new double[nHidden + 1];
				hiddenNeurons[nHidden] = 1;

				outputNeurons = new double[nOutput];

				wInputHidden = new double[nHidden, nInput + 1];
                wHiddenOutput = new double[nOutput, nHidden + 1];

				for(int i = 0; i <= nInput; i++)
				{
					for(int j = 0; j < nHidden; j++)
					{
						wInputHidden[j, i] = reader.ReadDouble();
					}
				}
				for(int i = 0; i <= nHidden; i++)
				{
					for(int j = 0; j < nOutput; j++)
					{
						wHiddenOutput[j, i] = reader.ReadDouble();
					}
				}
			}}
		}

        public void CopyFrom(vnn target)
        {
            if(target.nInput != this.nInput || target.nHidden != this.nHidden || target.nOutput != this.nOutput) { throw new Exception("Nets are not of same dimensions!"); }

            Array.Copy(target.wInputHidden, this.wInputHidden, wInputHidden.Length);
            Array.Copy(target.wHiddenOutput, this.wInputHidden, wHiddenOutput.Length);
        }
        public vnn Copy()
        {
            var wih = new double[nHidden, nInput + 1];
            var who = new double[nOutput, nHidden + 1];
            Array.Copy(wInputHidden, wih, wInputHidden.Length);
            Array.Copy(wHiddenOutput, who, wHiddenOutput.Length);

            return new vnn(wih, who);
        }
        public vnn CopyWithNeurons()
        {
            var copy = Copy();

            Array.Copy(inputNeurons, copy.inputNeurons, inputNeurons.Length);
            Array.Copy(hiddenNeurons, copy.hiddenNeurons, hiddenNeurons.Length);
            Array.Copy(outputNeurons, copy.outputNeurons, outputNeurons.Length);

            return copy;
        }

		static void printNeurons(System.Text.StringBuilder b, double[] neurons, string name) {
			b.AppendLine($"N-{name}: ");
			for(int y = 0, toy = neurons.Length; y < toy; y++){
				b.Append(neurons[y].ToString("N2").PadLeft(6));
			}
			b.AppendLine();
		}
		static void printWeights(System.Text.StringBuilder b, double[,] w, string ins, string outs) {
			b.AppendLine($"{ins} x {outs}: ");
			for(int x = 0, tox = w.GetLength(0), toy = w.GetLength(1); x < tox; x++){
				for(int y = 0; y < toy; y++){
					b.Append(w[x, y].ToString("N2").PadLeft(6));
				}
				b.AppendLine();
			}
			b.AppendLine();
		}
		public string ToString(bool neurons) {
			var b = new System.Text.StringBuilder();            
            if(neurons){
				printNeurons(b, inputNeurons, "Inputs");
				printNeurons(b, hiddenNeurons, "Hidden");
				printNeurons(b, outputNeurons, "Output");
            }
            else {
				printWeights(b, wInputHidden, "Input", "Hidden");
				printWeights(b, wHiddenOutput, "Hidden", "Output");
			}   
            return b.ToString();
		}
		public override string ToString() { return ToString(false); }
    }
}
