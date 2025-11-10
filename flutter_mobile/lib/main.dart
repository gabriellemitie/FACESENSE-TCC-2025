import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
// Use native MethodChannel bridge (FlexBridge) for model loading and inference
// to avoid depending on the tflite_flutter plugin (which may use the old
// Android v1 embedding in some versions). The native bridge loads the
// Interpreter via reflection and attaches the Flex delegate when available.

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _nativeModelLoaded = false;
  String _status = 'Idle';
  static const platform = MethodChannel('com.facesense.mobile/flex_bridge');

  @override
  void initState() {
    super.initState();
    // Use the native bridge to load the model (avoids plugin embedding issues).
    _nativeLoadModel();
  }

  // We intentionally don't use the Dart tflite plugin here. Instead we call
  // the native FlexBridge (Kotlin) that loads the Interpreter and attaches
  // Flex via reflection. This avoids pub dependency conflicts and Android v1
  // embedding checks.

  Future<void> _nativeLoadModel() async {
    try {
      setState(() => _status = 'Calling native loadModel...');
      final res = await platform.invokeMethod('loadModel', {'asset': 'assets/best_model_select.tflite'});
      setState(() {
        _status = 'Native loadModel: $res';
        _nativeModelLoaded = true;
      });
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
    // Prefer the native bridge for inference. It will error if the native
    // interpreter hasn't been loaded.
    await _nativeRunInference();
  }

  @override
  void dispose() {
    // The native interpreter will be cleaned up by the Android process when
    // the app terminates; no Dart-side interpreter to close when using the
    // native bridge.
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
                child: const Text('Run dummy inference (native)'),
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
