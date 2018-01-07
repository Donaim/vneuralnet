using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Windows.Forms;
using System.Drawing;

using VNNAddOn;
using VNNLib;

namespace visualizer
{
	static class Program
	{
		static void Main(string[] args)
		{
			root.nnh = new root.NNhandler();
			
			var con = new root.controller();
			root.mform = new root.form();
			root.mform.Show();
			con.Show();
			con.Focus();
			con.Activate();
			Application.Run(con);
		}
	}
	partial class root
	{
		static readonly Random rand = new Random();
		static Rectangle windowBoarders;
		public static NNhandler nnh;
		public static form mform;
		public class form : Form
		{
			BufferedGraphics gfx;
			System.Windows.Forms.Timer timer = new Timer();
			protected override Size DefaultSize { get; } = new Size(500, Screen.PrimaryScreen.WorkingArea.Height);
			public form()
			{
				FormBorderStyle = FormBorderStyle.None;
				StartPosition = FormStartPosition.Manual;
				Location = new Point(0, 0);

				Graphics g = CreateGraphics();
				windowBoarders = new Rectangle((int)g.VisibleClipBounds.X, (int)g.VisibleClipBounds.Y, (int)g.VisibleClipBounds.Width, (int)g.VisibleClipBounds.Height);

				gfx = BufferedGraphicsManager.Current.Allocate(CreateGraphics(), ClientRectangle);
				gfx.Graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceOver;
				gfx.Graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
				gfx.Graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
				gfx.Graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;
				gfx.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

				nnh.init_graphics(gfx);

				timer.Enabled = true;
				timer.Interval = 1000 / 24;
				timer.Tick += nnh.timer_draw;
			}

			protected override void OnPaint(PaintEventArgs e) { }
			protected override void OnPaintBackground(PaintEventArgs e) { }

		}
		public class NNhandler
		{
			BufferedGraphics gfx;
			public NNhandler()
			{
				nn = new vnn(2, 5, 1);
				tr = new trainer(nn, data.DataSets.xorProblem, 0.0001, 0.9);
				rep = new reporter(nn);
			}
			public void init_graphics(BufferedGraphics g)
			{
				gfx = g;
				generate_points();
			}
			const int 
				boarder_shift = 100,
				ball_size = 50;
			void generate_points()
			{
				ip = new Point[nn.nInput + 1];
				ir = new Rectangle[nn.nInput + 1];
				double parts = windowBoarders.Height / (double)(nn.nInput + 1);
				for(int i = 0; i <= nn.nInput; i++)
				{
					ip[i] = new Point(boarder_shift - ball_size, (int)(parts / 2 + i * parts));
					ir[i] = new Rectangle(ip[i].X - ball_size / 2, ip[i].Y - ball_size / 2, ball_size, ball_size);
				}

				hp = new Point[nn.nHidden + 1];
				hr = new Rectangle[nn.nHidden + 1];
				parts = windowBoarders.Height / (double)(nn.nHidden + 1);
				for(int i = 0; i <= nn.nHidden; i++)
				{
					hp[i] = new Point(windowBoarders.Width / 2 - ball_size / 2, (int)(parts / 2 + i * parts));
					hr[i] = new Rectangle(hp[i].X - ball_size / 2, hp[i].Y - ball_size / 2, ball_size, ball_size);
				}

				op = new Point[nn.nOutput];
				or = new Rectangle[nn.nOutput];
				parts = windowBoarders.Height / (double)(nn.nOutput);
				for(int i = 0; i < nn.nOutput; i++)
				{
					op[i] = new Point(windowBoarders.Width - boarder_shift, (int)(parts / 2 + i * parts));
					or[i] = new Rectangle(op[i].X - ball_size / 2, op[i].Y - ball_size / 2, ball_size, ball_size);
				}
			}
			
			class ball : Control
			{
				Rectangle rectangle, selrec;
				public readonly int CenterX, CenterY;
				public ball(int cx, int cy, Rectangle rec)
				{
					DoubleBuffered = true;

					CenterX = cx;
					CenterY = cy;
					rectangle = rec;
					selrec = new Rectangle(rec.X - 2, rec.Y - 2, rec.Width + 4, rec.Height + 4);

					KeyDown += Ball_KeyDown;
					MouseEnter += Ball_MouseEnter;
					MouseLeave += Ball_MouseLeave;
				}

				bool mouseon = false;
				private void Ball_MouseEnter(object sender, EventArgs e)
				{
					mouseon = true;
				}
				private void Ball_MouseLeave(object sender, EventArgs e)
				{
					mouseon = false;
				}

				private void Ball_KeyDown(object sender, KeyEventArgs e)
				{
					double d;
					if(e.KeyCode == Keys.Back)
					{
						if(text.Length > 0) { text = text.Remove(text.Length - 1); }
					}
					else if(double.TryParse(new string(new char[] { (char)e.KeyValue }), out d))
					{
						text += (char)e.KeyValue;
					}
				}

				string text = string.Empty;
				static readonly Font font = new Font("Consolas", 9f);
				static readonly Brush back = Brushes.Blue;
				public void DrawMe(BufferedGraphics gfx)
				{
					if(mouseon) { gfx.Graphics.FillEllipse(Brushes.Yellow, selrec); }
					gfx.Graphics.FillEllipse(back, rectangle);
					gfx.Graphics.DrawString(text, font, Brushes.White, 10f, 10f);
				}
				protected override void OnPaint(PaintEventArgs e) { }
				protected override void OnPaintBackground(PaintEventArgs pevent) { }
			}
			Point[] ip, hp, op;
			Rectangle[] ir, hr, or;
			static readonly Color back = Color.FromArgb(200, 250, 200);
			public void timer_draw(object sender, EventArgs e)
			{
				gfx.Graphics.Clear(back);

				double[,] wih = new double[nn.nInput + 1, nn.nHidden], who = new double[nn.nHidden + 1, nn.nOutput];

				double imin = double.MaxValue, imax = double.MinValue;
				for(int i = 0; i <= nn.nInput; i++)
				{
					for(int j = 0; j < nn.nHidden; j++)
					{
						wih[i, j] = Math.Abs(nn.wInputHidden[i, j]);
						imin = Math.Min(imin, wih[i, j]);
						imax = Math.Max(imax, wih[i, j]);
					}
				}
				double imult = 255 / (imax - imin);
				double hmin = double.MaxValue, hmax = double.MinValue;
				for(int i = 0; i <= nn.nHidden; i++)
				{
					for(int j = 0; j < nn.nOutput; j++)
					{
						who[i, j] = Math.Abs(nn.wHiddenOutput[i, j]);
						hmin = Math.Min(hmin, who[i, j]);
						hmax = Math.Max(hmax, who[i, j]);
					}
				}
				double hmult = 255 / (hmax - hmin);

				for(int i = 0; i <= nn.nInput; i++)
				{
					for(int j = 0; j < nn.nHidden; j++)
					{
						int col = (int)(imult * (wih[i, j] - imin));
						gfx.Graphics.DrawLine(new Pen(Color.FromArgb(col, col, col), 5), ip[i], hp[j]);
					}
				}
				for(int i = 0; i <= nn.nHidden; i++)
				{
					for(int j = 0; j < nn.nOutput; j++)
					{
						int col = (int)(hmult * (who[i, j] - hmin));
						gfx.Graphics.DrawLine(new Pen(Color.FromArgb(col, col, col), 5), hp[i], op[j]);
					}
				}

				for(int i = 0; i <= nn.nInput; i++)
				{
					gfx.Graphics.FillEllipse(Brushes.Blue, ir[i]);
				}
				for(int i = 0; i <= nn.nHidden; i++)
				{
					gfx.Graphics.FillEllipse(Brushes.Blue, hr[i]);
				}
				for(int i = 0; i < nn.nOutput; i++)
				{
					gfx.Graphics.FillEllipse(Brushes.Blue, or[i]);
				}

				gfx.Render();
			}
		}
		static vnn nn;
		static trainer tr;
		static reporter rep;
		public class controller : Form
		{
			protected override Size DefaultSize { get; } = new Size(300, 200);
			public controller()
			{
				StartPosition = FormStartPosition.CenterScreen;
				KeyPreview = true;

				KeyDown += Controller_KeyDown;
				KeyUp += Controller_KeyUp;

				boxes = new TextBox[nn.nInput];
				for(int i = 0; i < nn.nInput; i++)
				{
					boxes[i] = new TextBox() { Size = new Size(50, 20), Location = new Point(i * (50 + 5) + 10, 20) };
					Controls.Add(boxes[i]);
				}
			}
			TextBox[] boxes;

			bool working = false;
			private void Controller_KeyUp(object sender, KeyEventArgs e)
			{
				if(e.KeyCode == Keys.Space)
				{
					working = !working;
					if(working)
					{
						Task.Factory.StartNew(working_void);
					}
				}
				else if(e.KeyCode == Keys.Enter)
				{
					
				}
			}
			void working_void()
			{
				while(working)
				{
					tr.TrainEpoch();
				}
			}

			public int interval = 1;
				private void Controller_KeyDown(object sender, KeyEventArgs e)
			{
				if(e.KeyCode == Keys.Right)
				{
					Task.Factory.StartNew(() => tr.TrainEpoch());
				}
			}
		}
	}
}
