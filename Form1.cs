using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using HttpClientProgress;
using System.Net.Http;

namespace testOpenCVSharpDnn
{
    public partial class Form1 : Form
    {
        private const string cfgFile = "yolov3.cfg";
        private const string cfgURI = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg";
        private const string weightsFile = "yolov3.weights";
        private const string weightsURI = "https://pjreddie.com/media/files/yolov3.weights";
        private const string labelsFile = "imagenet.shortnames.list";
        private const string labelsURI = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/imagenet.shortnames.list";
        private const string sampleImage = "kite.jpg";
        private const string sampleImageURI = "https://github.com/pjreddie/darknet/raw/master/data/kite.jpg";
        private string[] Labels;
        const float threshold = 0.2f;
        const float nmsThreshold = 0.1f;
        private Net net;
        private static readonly HttpClient httpClient = new HttpClient();
        public Form1()
        {
            InitializeComponent();
            label1.Text = "";
            ChkFiles();
        }

        // サンプル用のファイルが無ければ拾ってくるためのボタンを有効にするのと
        // Cv2.GetBuildInformationをTextBoxに表示するだけ。大したことはやってない
        private void ChkFiles()
        {
            if (File.Exists(labelsFile) && File.Exists(cfgFile) && File.Exists(weightsFile) && File.Exists(sampleImage))
            {
                progressBar1.Enabled = false;
                button1.Enabled = false;
                button2.Enabled = true;
                net = CvDnn.ReadNetFromDarknet(cfgFile, weightsFile);
                // OPENCV_DNN_CUDAとWITH_CUDAを有効にしたOpenCVSharpだとCUDAを使ってくれる(x64)
                // ダメな場合は勝手にCPUに切り替えてくれる。
                net.SetPreferableBackend(Backend.CUDA);
                net.SetPreferableTarget(Target.CUDA_FP16);
                textBox1.Text = Cv2.GetBuildInformation().Replace("\n","\r\n");
                
            }
            else
            {
                button1.Enabled = true;
                button2.Enabled = false;
            }
        }

        // CvDnnでのYoloについては
        // https://github.com/died/OpenCvSharpDnnYolo をほぼ流用
        // Row[i]→Row(i)に変更してる
        private void button2_Click(object sender, EventArgs e)
        {
            Labels = File.ReadAllLines(labelsFile).ToArray();
            var org = new Mat(sampleImage);
            var blob = CvDnn.BlobFromImage(org, 1.0 / 255, new OpenCvSharp.Size(416, 416), new Scalar(), true, false);
            net.SetInput(blob);
            var outNames = net.GetUnconnectedOutLayersNames();
            var outs = outNames.Select(_ => new Mat()).ToArray();
            net.Forward(outs, outNames);
            int prefix = 5;
            var classIds = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect2d>();
            var w = org.Width;
            var h = org.Height;
            foreach (var prob in outs)
            {
                for (var i = 0; i < prob.Rows; i++)
                {
                    var confidence = prob.At<float>(i, 4);
                    if (confidence > threshold)
                    {
                        Cv2.MinMaxLoc(prob.Row(i).ColRange(prefix, prob.Cols), out _, out OpenCvSharp.Point max);
                        var classes = max.X;
                        var probability = prob.At<float>(i, classes + prefix);
                        if (probability > threshold)
                        {
                            var centerX = prob.At<float>(i, 0) * w;
                            var centerY = prob.At<float>(i, 1) * h;
                            var width = prob.At<float>(i, 2) * w;
                            var height = prob.At<float>(i, 3) * h;
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(centerX, centerY, width, height));

                        }

                    }

                }
            }
            CvDnn.NMSBoxes(boxes, confidences, threshold, nmsThreshold, out int[] indices);
            Console.WriteLine($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");

            // ここはFormsアプリに合わせて作成
            Image image = Image.FromFile(sampleImage);
            Pen pen = new Pen(Color.Red, 10.0F);
            Font fnt = new Font("MS Gothic", 12);

            using (var canvas = Graphics.FromImage(image))
            {
                foreach (var i in indices)
                {
                    float x = (float)boxes[i].X;
                    float y = (float)boxes[i].Y;
                    float width = (float)boxes[i].Width;
                    float height = (float)boxes[i].Height;
                    float x1 = (x - width / 2) < 0 ? 0 : x - width / 2;
                    float y1 = (y - height / 2);
                    canvas.DrawRectangle(pen, x1, y1, width, height);
                    canvas.DrawString(Labels[classIds[i]], fnt, Brushes.Red, x, y + image.Height / 40);
                    canvas.DrawString(string.Format("{0:0.00}", confidences[i]), fnt, Brushes.Red, x, y + image.Height / 15);
                    canvas.Flush();
                    Console.WriteLine($"{i}: {Labels[classIds[i]]}, {confidences[i]}, {probabilities[i]}");

                }

            }
            pictureBox1.Image = image;
        }

        // WebClientでダウンロードをさせたが画面が固まるし廃止らしいのでHttpClientを使う方法に変えたが
        // どうせなら進捗プログレスバーが欲しかったがUWPより面倒そうだったので他所から拝借
        private async void button1_Click(object sender, EventArgs e)
        {
            if (!File.Exists(labelsFile))
            {
                progressBar1.Value = 0;
                label1.Text = labelsFile;
                await DownloadFile(labelsURI, labelsFile);
                
            }
            if (!File.Exists(cfgFile))
            {
                progressBar1.Value = 0;
                label1.Text = cfgFile;
                await DownloadFile(cfgURI, cfgFile);
            }
            if (!File.Exists(weightsFile))
            {
                progressBar1.Value = 0;
                label1.Text = weightsFile;
                await DownloadFile(weightsURI, weightsFile);
            }
            if (!File.Exists(sampleImage))
            {
                progressBar1.Value = 0;
                label1.Text = sampleImage;
                await DownloadFile(sampleImageURI, sampleImage);
            }
            ChkFiles();
        }

        // まんまやりたい事が載ってた
        // https://gist.github.com/dalexsoto/9fd3c5bdbe9f61a717d47c5843384d11
        // 進捗表示用プログレスバーを追加
        private async Task DownloadFile(string URI, string filename)
        {
            var progress = new Progress<float>();
            progress.ProgressChanged += Progress_ProgressChanged;

            using (var file = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await httpClient.DownloadDataAsync(URI, file, progress);
            }
        }

        void Progress_ProgressChanged(object sender, float progress)
        {
            // Do something with your progress
            progressBar1.Value = (int)progress;
            Console.WriteLine(progress);
        }
    }
}
