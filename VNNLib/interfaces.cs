namespace VNNLib {
    public interface IFeedForwardNN
    {
        void feedForward(double[] pattern);
    }
    public interface IFeedResultNN
    {
        double[] feedResult(double[] pattern);
    }
    public interface ISimpleMLP
    {
        double[,] GwInputHidden { get; }
        double[,] GwHiddenOutput { get; }
        double[] GInputNeurons { get; }
        double[] GHiddenNeurons { get; }
        double[] GOutputNeurons { get; }

        int NInput { get; }
        int NHidden { get; }
        int NOutput { get; }
    }
    public interface ICopyableNN<T> {
        void CopyFrom(T target);
        T Copy();
        T CopyWithNeurons();
    }

    
    public interface ITrainer
    {
        void backpropagate(double[] desiredOutputs, double learningRate);
    }

    public delegate double RandomizeFunc(int n_in, int n_out, int i);
}