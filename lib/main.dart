import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'dart:math';

void main() {
  runApp(const FaceRecognitionApp());
}

class FaceRecognitionApp extends StatelessWidget {
  const FaceRecognitionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: const Color(0xFF6366F1), // Indigo
        useMaterial3: true,
      ),
      home: const FaceRecognitionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class FaceRecognitionScreen extends StatefulWidget {
  const FaceRecognitionScreen({super.key});

  @override
  _FaceRecognitionScreenState createState() => _FaceRecognitionScreenState();
}

class _FaceRecognitionScreenState extends State<FaceRecognitionScreen> {
  Interpreter? _interpreter;
  late FaceDetector _faceDetector;
  File? _image1;
  File? _image2;
  List<double>? _embedding1;
  List<double>? _embedding2;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
    _faceDetector = FaceDetector(
        options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate));
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/facenet.tflite');
      print("✅ TFLite Model Loaded Successfully");
      
      var inputShape = _interpreter?.getInputTensors()[0].shape;
      var outputShape = _interpreter?.getOutputTensors()[0].shape;
      print("📊 Model Input: $inputShape, Output: $outputShape");
    } catch (e) {
      print("❌ Failed to load model: $e");
    }
  }

  Future<void> _pickImage(int imageNumber, ImageSource source) async {
    final pickedFile = await ImagePicker().pickImage(
      source: source,
      imageQuality: 50,
    );
    if (pickedFile == null) return;

    setState(() {
      if (imageNumber == 1) {
        _image1 = File(pickedFile.path);
        _embedding1 = null;
      } else {
        _image2 = File(pickedFile.path);
        _embedding2 = null;
      }
      _isProcessing = true;
    });

    await _processImage(imageNumber);
    
    setState(() {
      _isProcessing = false;
    });
  }

  Future<void> _processImage(int imageNumber) async {
    if (_interpreter == null) {
      _showError("TFLite Model not loaded yet.");
      return;
    }

    File? imageFile = (imageNumber == 1) ? _image1 : _image2;
    if (imageFile == null) return;

    try {
      Uint8List imageBytes = await imageFile.readAsBytes();
      
      // 1. Detect Face
      final inputImage = InputImage.fromFilePath(imageFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        throw "No face detected in this image. Try a clearer photo.";
      }

      // 2. Decode & Crop
      img.Image? fullImage = img.decodeImage(imageBytes);
      if (fullImage == null) throw "Could not decode image.";

      final face = faces.first;
      img.Image croppedFace = img.copyCrop(
        fullImage,
        x: face.boundingBox.left.toInt(),
        y: face.boundingBox.top.toInt(),
        width: face.boundingBox.width.toInt(),
        height: face.boundingBox.height.toInt(),
      );

      // 3. Prep Tensors
      var inputTensor = _interpreter!.getInputTensors()[0];
      var outputTensor = _interpreter!.getOutputTensors()[0];
      int inputSize = inputTensor.shape[1];
      int outputSize = outputTensor.shape[1];

      List<List<List<List<double>>>> input = _convertImageToTfList(croppedFace, inputSize);
      var output = List<double>.filled(outputSize, 0).reshape([1, outputSize]);

      // 4. Run AI
      _interpreter?.run(input, output);

      setState(() {
        if (imageNumber == 1) {
          _embedding1 = _l2Normalize(List<double>.from(output[0]));
        } else {
          _embedding2 = _l2Normalize(List<double>.from(output[0]));
        }
      });
    } catch (e) {
      _showError(e.toString());
    }
  }

  List<List<List<List<double>>>> _convertImageToTfList(img.Image image, int size) {
    img.Image resized = img.copyResize(image, width: size, height: size);
    List<List<List<List<double>>>> normalizedImage =
    List.generate(1, (_) => List.generate(size, (_) => List.generate(size, (_) => [0.0, 0.0, 0.0])));

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        img.Pixel pixel = resized.getPixel(j, i);
        normalizedImage[0][i][j] = [
          (pixel.r / 255.0 - 0.5) * 2.0,
          (pixel.g / 255.0 - 0.5) * 2.0,
          (pixel.b / 255.0 - 0.5) * 2.0,
        ];
      }
    }
    return normalizedImage;
  }

  List<double> _l2Normalize(List<double> embedding) {
    double norm = sqrt(embedding.fold(0.0, (sum, val) => sum + val * val));
    return norm == 0 ? embedding : embedding.map((v) => v / norm).toList();
  }

  double _calculateDistance(List<double> emb1, List<double> emb2) {
    double sum = 0.0;
    for (int i = 0; i < emb1.length; i++) {
      sum += pow(emb1[i] - emb2[i], 2);
    }
    return sqrt(sum);
  }

  void _compareFaces() {
    if (_embedding1 == null || _embedding2 == null) {
      _showError("Select both images first.");
      return;
    }

    double distance = _calculateDistance(_embedding1!, _embedding2!);
    double threshold = 0.85; // Stricter threshold to avoid false positives (0.8 - 0.9 range)
    bool isMatch = distance < threshold;
    double similarity = max(0, 100 - (distance * 50)); // Map distance [0, 2] to % [100, 0]

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Color(0xFF1E1E2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isMatch ? Icons.check_circle_outline : Icons.error_outline,
              color: isMatch ? Colors.greenAccent : Colors.redAccent,
              size: 72,
            ),
            const SizedBox(height: 16),
            Text(
              isMatch ? "Verified Match" : "Faces Don't Match",
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              "Similarity: ${similarity.toStringAsFixed(1)}%",
              style: TextStyle(color: isMatch ? Colors.greenAccent : Colors.redAccent, fontSize: 18, fontWeight: FontWeight.w600),
            ),
            Text(
              "Raw Distance: ${distance.toStringAsFixed(3)}",
              style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF6366F1),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                child: const Text("CLOSE"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: Colors.redAccent, behavior: SnackBarBehavior.floating),
    );
  }

  @override
  void dispose() {
    _interpreter?.close();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      body: Stack(
        children: [
          // Background Glow
          Positioned(
            top: -100,
            right: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF6366F1).withOpacity(0.1),
              ),
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 24),
                  const Text("Face", style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.white)),
                  const Text("Comparison AI", style: TextStyle(fontSize: 32, fontWeight: FontWeight.w300, color: Color(0xFF818CF8))),
                  const SizedBox(height: 32),
                  Expanded(
                    child: ListView(
                      physics: const BouncingScrollPhysics(),
                      children: [
                        _buildImageSlot(1, _image1, "Primary ID"),
                        const SizedBox(height: 20),
                        _buildImageSlot(2, _image2, "Secondary Face"),
                        const SizedBox(height: 40),
                        Center(
                          child: GestureDetector(
                            onTap: _compareFaces,
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 300),
                              padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 20),
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  colors: (_embedding1 != null && _embedding2 != null)
                                      ? [const Color(0xFF6366F1), const Color(0xFF4F46E5)]
                                      : [Colors.grey.shade800, Colors.grey.shade900],
                                ),
                                borderRadius: BorderRadius.circular(20),
                                boxShadow: [
                                  if (_embedding1 != null && _embedding2 != null)
                                    BoxShadow(color: const Color(0xFF6366F1).withOpacity(0.4), blurRadius: 20, offset: const Offset(0, 10))
                                ],
                              ),
                              child: const Text("START ANALYSIS", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
                            ),
                          ),
                        ),
                        const SizedBox(height: 40),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (_isProcessing)
            Container(
              color: Colors.black54,
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 4, sigmaY: 4),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: Color(0xFF6366F1)),
                      SizedBox(height: 20),
                      Text("Analyzing Features...", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w500)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildImageSlot(int num, File? file, String label) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 13, fontWeight: FontWeight.w600, letterSpacing: 1)),
        const SizedBox(height: 8),
        Container(
          height: 200,
          width: double.infinity,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.03),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Colors.white.withOpacity(0.08)),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: Stack(
              children: [
                if (file != null)
                  Image.file(file, width: double.infinity, height: 200, fit: BoxFit.cover)
                else
                  Center(child: Icon(Icons.face_retouching_natural, size: 48, color: Colors.white.withOpacity(0.1))),
                Positioned(
                  bottom: 12,
                  right: 12,
                  child: Row(
                    children: [
                      _buildMiniIconButton(Icons.photo_library, () => _pickImage(num, ImageSource.gallery)),
                      const SizedBox(width: 8),
                      _buildMiniIconButton(Icons.camera_alt, () => _pickImage(num, ImageSource.camera)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMiniIconButton(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFF1E1E2E).withOpacity(0.9),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withOpacity(0.1)),
        ),
        child: Icon(icon, size: 20, color: const Color(0xFF818CF8)),
      ),
    );
  }
}
