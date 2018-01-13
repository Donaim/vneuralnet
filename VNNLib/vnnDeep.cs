using static System.Math;

using System;

namespace VNNLib {
    public class vnnDeep {
        private readonly double [][,] L;
        private readonly int[] size;
        private double [][] N;
        public vnnDeep(double [][,] layers){
	        L = layers;

            size = new int[L.Length + 1];
            size[0] = L[0].GetLength(0);
            for(int i = 0; i < L.Length; i++) {
                size[i + 1] = L[i].GetLength(1);
            }

            zero_neurons();
        }
        void zero_neurons(){
            N = new double[size.Length][];
            for(int i = 0; i < N.Length - 1; i++){
                N[i] = new double[size[i] + 1];
                N[i][size[i]] = 1.0; //bias neuron
            }
            N[N.Length - 1] = new double[size[N.Length - 1]]; //last neuron layer does not have bias
        }
       
        public unsafe byte[] ToBytes()
		{
			using(var stream = new System.IO.MemoryStream()) {
			using(var writer = new System.IO.BinaryWriter(stream))
			{
                writer.Write(size.Length);
                for(int i = 0; i < size.Length; i++) { writer.Write(size[i]); }

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
                size = new int[lcount];
                for(int i = 0; i < lcount; i++) {
                    size[i] = reader.ReadInt32();
                }

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
                vnn.mult(L[i], N[i], N[i + 1], size[i], size[i + 1]);
            }

            // vnn.mult(wInputHidden, inputNeurons, hiddenNeurons, nInput, nHidden);
            // vnn.mult(wInputHidden, inputNeurons, hiddenNeurons, nInput, nHidden);
            // vnn.mult(wHiddenOutput, hiddenNeurons, outputNeurons, nHidden, nOutput);
      
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

        public static vnnDeep CreateEmpty(params int[] sizes) {
            var l = new double[sizes.Length - 1][,];
            for(int i = 0; i < sizes.Length - 1; i++) {
                l[i] = new double[sizes[i], sizes[i + 1]];
            }

            return new vnnDeep(l);
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