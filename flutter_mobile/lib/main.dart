import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  Interpreter? _interpreter;
  String _status = 'Idle';
  static const platform = MethodChannel('com.facesense.mobile/flex_bridge');

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() => _status = 'Loading model...');
    try {
      // Attempt to load the tflite model included in assets
      final interpreterOptions = InterpreterOptions();
      // NOTE: If the model requires Flex/Select TF ops, you may need to
      // include the Flex delegate native libraries in your Android/iOS build
      // and create a Delegate using native code. tflite_flutter does not
      // directly expose a Flex delegate factory that works without adding
      // native support. See README.md for packaging instructions.

      _interpreter = await Interpreter.fromAsset(
        'assets/best_model_select.tflite',
        options: interpreterOptions,
      );

      setState(() => _status = 'Model loaded');
    } catch (e, st) {
      setState(() => _status = 'Failed to load model: $e');
      // Keep the full exception in console for debugging
      // ignore: avoid_print
      print(e);
      // ignore: avoid_print
      print(st);
    }
  }

  Future<void> _nativeLoadModel() async {
    try {
      setState(() => _status = 'Calling native loadModel...');
      final res = await platform.invokeMethod('loadModel', {'asset': 'assets/best_model_select.tflite'});
      setState(() => _status = 'Native loadModel: $res');
    } on PlatformException catch (e) {
      setState(() => _status = 'Native loadModel failed: ${e.message}');
    }
  }

  Future<void> _nativeRunInference() async {
    try {
      setState(() => _status = 'Calling native runInference...');
      // Create dummy input arr flattened
      final input = List<double>.generate(200 * 99, (i) => (Random().nextDouble() * 2 - 1));
      final res = await platform.invokeMethod('runInference', {'input': input});
      setState(() => _status = 'Native inference result: ${res.toString().substring(0, min(200, res.toString().length))}');
    } on PlatformException catch (e) {
      setState(() => _status = 'Native runInference failed: ${e.message}');
    }
  }

  Future<void> _runDummyInference() async {
    if (_interpreter == null) {
      setState(() => _status = 'Interpreter not loaded');
      return;
    }

    try {
      setState(() => _status = 'Running inference...');

      // The original model expects shape (1, 200, 99) -> output (1, 24)
      // Build a dummy input tensor with random floats to verify interpreter works
      final inputShape = [1, 200, 99];
      final outputShape = [1, 24];

      // Create input buffer
      final input = TensorBuffer.createFixedSize(List<int>.from(inputShape), TfLiteType.float32);
      final rnd = Random();
      final inputData = List.generate(200 * 99, (_) => (rnd.nextDouble() * 2 - 1).toDouble());
      input.loadList(inputData, shape: inputShape);

      final output = TensorBuffer.createFixedSize(List<int>.from(outputShape), TfLiteType.float32);

      // Run inference
      _interpreter!.run(input.buffer, output.buffer);

      setState(() => _status = 'Inference done, output len=${output.getDoubleList().length}');
    } catch (e, st) {
      setState(() => _status = 'Inference failed: $e');
      // ignore: avoid_print
      print(e);
      // ignore: avoid_print
      print(st);
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('FaceSense Mobile (PoC)')),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Status: $_status'),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _runDummyInference,
                child: const Text('Run dummy inference'),
              ),
              const SizedBox(height: 12),
              const Text('Notes:'),
              const SizedBox(height: 6),
              const Text('- This app demonstrates loading the select-tflite model.'),
              const Text('- To run the real model on-device you must bundle Flex delegate (see README).'),
            ],
          ),
        ),
      ),
    );
  }
}
