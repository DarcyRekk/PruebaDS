using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;

class Program
{
    static void Main()
    {
        string videoPath = "/VideoEntrada.mp4";
        string outputFolder = "/SalidaImagen";

        using (VideoCapture capture = new VideoCapture(videoPath))
        {
            if (!capture.IsOpened)
            {
                Console.WriteLine("Error al abrir el video.");
                return;
            }

            int frameRate = (int)capture.Get(CapProp.Fps);
            int targetFrameCount = frameRate * 10;  // 10 segundos de video
            int frameCount = 0;

            // Crea la carpeta de salida si no existe
            System.IO.Directory.CreateDirectory(outputFolder);

            while (frameCount < targetFrameCount)
            {
                Mat frame = new Mat();
                capture.Read(frame);

                if (frame.IsEmpty)
                    break;

                // Guarda cada frame como una imagen
                string imagePath = System.IO.Path.Combine(outputFolder, $"frame_{frameCount}.png");
                CvInvoke.Imwrite(imagePath, frame);

                using (CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml"))
                {
                    var grayFrame = new Mat();
                    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    Rectangle[] faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 5);

                    foreach (var face in faces)
                    {
                        // Analizar las diferencias faciales y generar la imagen correspondiente
                        Mat facialDifferences = AnalizarDiferenciasFaciales(frame, face);

                        // Guardar la imagen de las diferencias faciales
                        string differencesImagePath = System.IO.Path.Combine(outputFolder, $"differences_{frameCount}.png");
                        CvInvoke.Imwrite(differencesImagePath, facialDifferences);
                    }
                }

                frameCount++;
            }
            Console.WriteLine("Imagenes guardadas satisfactoriamente");
        }
    }
    static Mat AnalizarDiferenciasFaciales(Mat frame, Rectangle face)
    {
        // Extraer la región de la cara
        Mat faceRegion = new Mat(frame, face);
        // Convertir la región de la cara a escala de grises si es necesario
        if (faceRegion.NumberOfChannels > 1)
        {
            CvInvoke.CvtColor(faceRegion, faceRegion, ColorConversion.Bgr2Gray);
        }
        // Dibujar un rectángulo alrededor de la cara en el frame original
        CvInvoke.Rectangle(frame, face, new MCvScalar(0, 255, 0), 2);

        // Dibujar un rectángulo alrededor de la región de la cara en la imagen de diferencias
        CvInvoke.Rectangle(faceRegion, new Rectangle(0, 0, faceRegion.Width, faceRegion.Height), new MCvScalar(255, 0, 0), 2);

        // Combina el frame original y la región de la cara con diferencias resaltadas
        Mat combinedImage = new Mat();
        CvInvoke.AddWeighted(frame, 1.0, faceRegion, 0.7, 0.0, combinedImage);

        return combinedImage;
    }
}

