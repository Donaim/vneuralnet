using VNNLib;
using VNNAddOn;

namespace VNNAddOn.exceptions
{
	public class NNSetNotMatch : System.Exception
	{
		public NNSetNotMatch(bool inputs)
		{
			if(inputs)
			{
				mess = "Neural Network inputs lenght and Training Set pattern lenght are not the same!";
			}
			else
			{
				mess = "Neural Network outputs lenght and Training Set pattern lenght are not the same!";
			}
		}
		public static void check_ex(vnn nn, VNNAddOn.trainingSet tset)
		{
			if(nn.nInput != tset.inputs[0].Length) { throw new NNSetNotMatch(true); }
			if(nn.nOutput != tset.outputs[0].Length) { throw new NNSetNotMatch(false); }
		}
		readonly string mess;
		public override string Message { get { return mess; } }
	}
	public class TSetDifferentInputs : System.Exception
	{
		public TSetDifferentInputs() { }
		public override string Message
		{
			get
			{
				return "Inputs and Outputs arrays have different lenght!";
			}
		}
	}
}
