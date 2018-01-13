using static System.Math;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

using System;

namespace VNNLib {
    public class vnnDeep : IFeedForwardNN, IFeedResultNN {
        public readonly double [][,] L;
        public readonly IReadOnlyList<int> size;
        public double [][] N;
        public vnnDeep(double [][,] layers){
	        L = layers;

            var sizeinit = new int[L.Length + 1];
            for(int i = 0; i < L.Length; i++) {
                sizeinit[i] = L[i].GetLength(1);
            }
            sizeinit[L.Length] = L[L.Length - 1].GetLength(0);
            size = sizeinit;

            zero_neurons();
        }
        void zero_neurons(){
            N = new double[size.Count][];
            for(int i = 0; i < N.Length; i++){
                N[i] = new double[size[i]];
                N[i][size[i] - 1] = 1.0; //bias neuron
            }
            N[size.Count - 1][size[size.Count -1] - 1] = 0; // output neurons doesnt have bias
        }
       
        public unsafe byte[] ToBytes()
		{
			using(var stream = new System.IO.MemoryStream()) {
			using(var writer = new System.IO.BinaryWriter(stream))
			{
                writer.Write(size.Count);
                for(int i = 0; i < size.Count; i++) { writer.Write(size[i]); }

                for(int i = 0; i < this.L.Length; i++){
                    var l = this.L[i];
                    for(int x = 0, tox = l.GetLength(0), toy = l.GetLength(1); x < tox; x++){
                        for(int y = 0; y < toy; y++){
                            writer.Write(l[x, y]);
                        }
                    }
                }
			}
			return stream.ToArray();
            }
		}
        public vnnDeep(byte[] raw){
            using(var stream = new System.IO.MemoryStream(raw)) {
            using(var reader = new System.IO.BinaryReader(stream))
            {
                int lcount = reader.ReadInt32();
                var sizeinit = new int[lcount];
                for(int i = 0; i < lcount; i++) {
                    sizeinit[i] = reader.ReadInt32();
                }
                size = sizeinit;

                L = new double[lcount - 1][,];
                for(int i = 0; i < this.L.Length; i++){
                    L[i] = new double[size[i], size[i + 1]];
                    var l = this.L[i];
                    for(int x = 0, tox = l.GetLength(0), toy = l.GetLength(1); x < tox; x++){
                        for(int y = 0; y < toy; y++){
                            l[x, y] = reader.ReadDouble();
                        }
                    }
                }
            }}

            zero_neurons();
        }

        public void feedForward(double[] pattern){
            Array.Copy(pattern, N[0], pattern.Length);

            for(int i = 0; i < L.Length; i++) {
                vnn.mult(L[i], N[i], N[i + 1]);
            }

            // vnn.mult(wInputHidden, inputNeurons, hiddenNeurons, nInput, nHidden);
            // vnn.mult(wInputHidden, inputNeurons, hiddenNeurons, nInput, nHidden);
            // vnn.mult(wHiddenOutput, hiddenNeurons, outputNeurons, nHidden, nOutput);
      
        }
        public double[] feedResult(double[] pattern) {
            feedForward(pattern);

			//create copy of output results
            int nOutput = size[size.Count - 1];
			double[] results = new double[nOutput];
            Array.Copy(N[size.Count - 1], 0, results, 0, nOutput);

			return results;
        }
        
        public string ToString(bool neurons){
            var b = new System.Text.StringBuilder();            
            if(neurons){
                for(int i = 0; i < this.N.Length; i++){
                    b.AppendLine($"N-{getLayerName(i)}: ");

                    var l = this.N[i];
                    for(int y = 0, toy = l.Length; y < toy; y++){
                        b.Append(l[y].ToString("N2").PadLeft(6));
                    }
                    b.AppendLine();
                }
            }
            else {
                for(int i = 0; i < this.L.Length; i++){
                    b.AppendLine($"{getLayerName(i)} x {getLayerName(i + 1)}: ");

                    var l = this.L[i];
                    for(int x = 0, tox = l.GetLength(0), toy = l.GetLength(1); x < tox; x++){
                        for(int y = 0; y < toy; y++){
                            b.Append(l[x, y].ToString("N2").PadLeft(6));
                        }
                        b.AppendLine();
                    }
                }   
            }
            return b.ToString();
        }
        public override string ToString() { return ToString(false); }
        string getLayerName(int i) {
            if(i == 0) { return "Inputs"; }
            else if (i >= this.L.Length) { return "Outputs"; }
            else { return $"Hidden[{i}]"; }
        }

        public vnn ToSimpleVNN(){
            if(size.Count != 3) { throw new Exception($"Cannot convert to simple vnn because size({size.Count}) is wrong"); }

            var re = new vnn(size[0] - 1, size[1] - 1, size[2]);

            for(int i = 0; i <= re.nInput; i++)
            {
                for(int j = 0; j < re.nHidden; j++)
                {
                    re.wInputHidden[j, i] = L[0][j, i];
                }
            }
            for(int i = 0; i <= re.nHidden; i++)
            {
                for(int j = 0; j < re.nOutput; j++)
                {
                    re.wHiddenOutput[j, i] = L[1][j, i];
                }
            }

            return re;
        }

        public static vnnDeep CreateEmpty(params int[] sizes) {
            var count = sizes.Length;
            var w = new double[count - 1][,];
            for(int i = 0; i < count - 1; i++) { // last is different
                w[i] = new double[sizes[i + 1], sizes[i] + 1]; // transposed & one bias 
            }
            return new vnnDeep(w);
        }
        public static vnnDeep CreateRandom(RandomizeFunc randomizer, params int[] sizes) {
            var re = CreateEmpty(sizes);

            for(int i = 0; i < re.L.Length; i++){
                var l = re.L[i];
                for(int x = 0, tox = l.GetLength(0), toy = l.GetLength(1); x < tox; x++){
                    for(int y = 0; y < toy; y++){
                        l[x, y] = randomizer(tox, toy, i);
                    }
                }
            }    

            return re;
        }
    }
}