import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math';

void main() {
  runApp(FaceRecognitionApp());
}

class FaceRecognitionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: FaceRecognitionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class FaceRecognitionScreen extends StatefulWidget {
  @override
  _FaceRecognitionScreenState createState() => _FaceRecognitionScreenState();
}

class _FaceRecognitionScreenState extends State<FaceRecognitionScreen> {
  Interpreter? _interpreter;
  File? _image1;
  File? _image2;
  List<double>? _embedding1;
  List<double>? _embedding2;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/facenet.tflite');
      print("✅ TFLite Model Loaded Successfully");
    } catch (e) {
      print("❌ Failed to load model: $e");
    }
  }


  Future<void> _pickImage(int imageNumber,int isCam) async {
    final pickedFile = await ImagePicker().pickImage(source:isCam==1? ImageSource.gallery: ImageSource.camera);
    if (pickedFile == null) return;

    setState(() {
      if (imageNumber == 1) {
        _image1 = File(pickedFile.path);
        _embedding1 = null;
      } else {
        _image2 = File(pickedFile.path);
        _embedding2 = null;
      }
    });

    _processImage(imageNumber);
  }

  Future<void> _processImage(int imageNumber) async {
    if (_interpreter == null) return;

    File? image = (imageNumber == 1) ? _image1 : _image2;
    if (image == null) return;

    Uint8List imageBytes = await image.readAsBytes();

    // Resize and Normalize the Image (FaceNet expects 160x160 input)
    List<List<List<List<double>>>> input = preprocessImage(imageBytes, 160, 160);

    List<List<double>> output = List.generate(1, (_) => List.filled(512, 0));
    _interpreter?.run(input, output);

    setState(() {
      if (imageNumber == 1) {
        _embedding1 = output[0];
      } else {
        _embedding2 = output[0];
      }
    });
  }

  List<List<List<List<double>>>> preprocessImage(Uint8List imageBytes, int width, int height) {
    // Decode the image using the image package
    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) {
      throw Exception("Failed to decode the image.");
    }

    // Resize image to the target dimensions (160x160)
    image = img.copyResize(image, width: width, height: height);

    // Create an array to store the normalized image
    List<List<List<List<double>>>> normalizedImage =
    List.generate(1, (_) => List.generate(height, (_) => List.generate(width, (_) => [0.0, 0.0, 0.0])));

    // Loop through all pixels and extract RGB values
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // Get the pixel at (x, y) coordinates
        img.Pixel pixel = image.getPixel(j, i);

        // Extract RGB components from the Pixel object
        num r = pixel.r;  // Red
        num g = pixel.g;  // Green
        num b = pixel.b;  // Blue

        // Normalize RGB values to [-1, 1]
        normalizedImage[0][i][j] = [
          (r / 255.0 - 0.5) * 2.0,  // Normalize red
          (g / 255.0 - 0.5) * 2.0,  // Normalize green
          (b / 255.0 - 0.5) * 2.0,  // Normalize blue
        ];
      }
    }

    return normalizedImage;
  }



  double _calculateDistance(List<double> emb1, List<double> emb2) {
    double sum = 0.0;
    for (int i = 0; i < emb1.length; i++) {
      sum += (emb1[i] - emb2[i]) * (emb1[i] - emb2[i]);
    }
    return sqrt(sum);
  }

  void _compareFaces() {
    if (_embedding1 != null && _embedding2 != null) {
      double distance = _calculateDistance(_embedding1!, _embedding2!);
      double threshold = 0.6; // Adjust for accuracy

      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text("Face Recognition Result"),
          content: Text(distance < threshold
              ? "✅ Faces Match (Distance: $distance)"
              : "❌ Faces Do Not Match (Distance: $distance)"),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: Text("OK"))
          ],
        ),
      );
    } else {
      print("⚠️ Please select both images first.");
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Face Recognition (TFLite)")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
           // Image.asset("assets/george.jpg"),
            _image1 != null ? Image.file(_image1!, height: 150) : Placeholder(fallbackHeight: 150),
            ElevatedButton(onPressed: () => _pickImage(1,1), child: Text("Pick First Image")),
            ElevatedButton(onPressed: () => _pickImage(1,2), child: Text("Pick First Image from camera")),
            _image2 != null ? Image.file(_image2!, height: 150) : Placeholder(fallbackHeight: 150),
            ElevatedButton(onPressed: () => _pickImage(2,1), child: Text("Pick Second Image")),
            ElevatedButton(onPressed: () => _pickImage(2,2), child: Text("Pick Second Image from camera")),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _compareFaces,
              child: Text("Compare Faces"),
            ),
          ],
        ),
      ),
    );
  }
}
