using System;
using System.Collections.Generic;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using System.IO;

namespace VNNLib
{
    public class vnnCm : IFeedResultNN
    {
        public vnnCm(double[][,] W, double[][] B, Func<double, double>[] activations)
        {
            Activations = activations;

            Weights = new Matrix<double>[W.Length];
            for(int i = 0; i < W.Length; i++)
            {
                Weights[i] = CreateMatrix.DenseOfArray(W[i]);
            }

            Biases = new Vector<double>[B.Length];
            for (int i = 0; i < B.Length; i++)
            {
                Biases[i] = CreateVector.DenseOfArray(B[i]);
            }

            Neurons = new Vector<double>[B.Length + 1];
            //Neurons[0] = CreateVector.DenseOfArray(new double[Weights[0].RowCount]);
            for (int i = 1; i < Neurons.Length; i++)
            {
                Neurons[i] = CreateVector.DenseOfArray(new double[Biases[i - 1].Count]);
            }
        }
        public vnnCm(double[][,] W, double[][] B, Func<double, double> activation) : this(W, B, CreateActivations(activation, B.Length)) { }
        static Func<double, double>[] CreateActivations(Func<double, double> single, int n) => new Func<double, double>[n].Select(z => single).ToArray();

        public readonly Matrix<double>[] Weights;
        public readonly Vector<double>[] Biases;

        public readonly Vector<double>[] Neurons;
        public readonly Func<double, double>[] Activations;

        public void feedForward(double[] pattern)
        {
            Neurons[0] = CreateVector.DenseOfArray(pattern);

            for(int l = 0, ln = Biases.Length; l < ln; l++)
            {
                Neurons[l + 1] = (Weights[l].LeftMultiply(Neurons[l])).Add(Biases[l]);
                Neurons[l + 1].MapInplace(Activations[l]);
            }
        }
        public double[] feedResult(double[] pattern)
        {
            feedForward(pattern);
            return Neurons[Biases.Length].ToArray();
        }


        public static Func<double, double> ParseFunc(string name)
        {
            switch (name.ToLower())
            {
                case "tanh": return Trig.Tanh;
                case "sigmoid": return MathNet.Numerics.SpecialFunctions.Logistic;
                case "relu": return (x) => x > 0 ? x : 0;
                case "linear": return (x) => x;

                default: throw new NotImplementedException();
            }
        }
        public static vnnCm LoadTxt(string dir)
        {
            var files = new DirectoryInfo(dir).GetFiles();
            List<double[,]> W = new List<double[,]>();
            List<double[]> B = new List<double[]>();
            List<Func<double, double>> A = new List<Func<double, double>>();

            foreach (var f in files)
            {
                var split = f.Name.Split('.');
                int index = int.Parse(split[0]);
                string tp = split[1];

                switch (tp)
                {
                    case "W": W.Add(parseWeights(f.FullName)); break;
                    case "B": B.Add(parseBiases(f.FullName)); break;
                    case "A": A.Add(ParseFunc(File.ReadAllText(f.FullName))); break;

                    default: continue;
                }
            }

            return new vnnCm(W.ToArray(), B.ToArray(), A.ToArray());

            double[,] parseWeights(string file)
            {
                var lines = File.ReadAllLines(file);
                var colN = lines[0].Split().Length;

                var re = new double[lines.Length, colN];
                for (int x = 0; x < lines.Length; x++)
                {
                    var sline = lines[x].Split(' ', ',', '\n', '\t');
                    for (int y = 0; y < colN; y++)
                    {
                        if (sline[y].Length == 0) { continue; }
                        re[x, y] = double.Parse(sline[y]);
                    }
                }

                return re;
            }
            double[] parseBiases(string file)
            {
                var split = File.ReadAllText(file).Split(' ', ',', '\t', '\n').Where(z => !string.IsNullOrWhiteSpace(z)).ToArray();
                var re = new double[split.Length];
                for (int i = 0; i < split.Length; i++)
                {
                    if (split[i].Length == 0) { continue; }
                    re[i] = double.Parse(split[i]);
                }
                return re;
            }
        }
    }
}
