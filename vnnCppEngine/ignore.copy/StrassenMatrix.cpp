struct Matrix{
public:
	int x;
	int y;
	double** matrix;
	
	Matrix(int _x, int _y)
	{
		x = _x;
		y = _y;
		
		matrix = new double*[x];
		for(int i = 0; i < x; i++){
			matrix[i] = new double[y];
		}
	}
};
