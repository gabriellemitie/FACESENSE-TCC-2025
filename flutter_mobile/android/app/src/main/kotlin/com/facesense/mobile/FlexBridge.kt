package com.facesense.mobile

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class FlexBridge(private val context: Context, private val channel: MethodChannel) : MethodChannel.MethodCallHandler {
    private var interpreter: Any? = null
    private val TAG = "FlexBridge"

    init {
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "loadModel" -> {
                val assetPath = call.argument<String>("asset") ?: run {
                    result.error("invalid_args", "asset path missing", null); return
                }
                try {
                    val file = copyAssetToFile(assetPath)
                    // Load interpreter using reflection to avoid hard dependency on the Flex API
                    try {
                        val interpreterClass = Class.forName("org.tensorflow.lite.Interpreter")
                        val optionsClass = Class.forName("org.tensorflow.lite.Interpreter\$Options")
                        val options = optionsClass.getDeclaredConstructor().newInstance()

                        // Try to instantiate FlexDelegate if available
                        try {
                            val flexClass = Class.forName("org.tensorflow.lite.flex.FlexDelegate")
                            val flexDelegate = flexClass.getDeclaredConstructor().newInstance()
                            val addDelegate = optionsClass.getMethod("addDelegate", Class.forName("org.tensorflow.lite.Delegate"))
                            addDelegate.invoke(options, flexDelegate)
                            Log.i(TAG, "FlexDelegate attached via reflection")
                        } catch (e: Exception) {
                            Log.w(TAG, "FlexDelegate not available: ${e.message}")
                        }

                        val ctor = interpreterClass.getConstructor(File::class.java, optionsClass)
                        val interp = ctor.newInstance(file, options)
                        interpreter = interp
                        result.success("ok")
                    } catch (e: Exception) {
                        Log.e(TAG, "Interpreter load failed: ${e.message}")
                        result.error("interp_error", "Interpreter load failed: ${e.message}", null)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "copy asset failed: ${e.message}")
                    result.error("copy_error", "copy asset failed: ${e.message}", null)
                }
            }
            "runInference" -> {
                try {
                    if (interpreter == null) {
                        result.error("no_model", "Interpreter not loaded", null); return
                    }
                    val inputList = call.argument<List<Double>>("input")
                    if (inputList == null) { result.error("invalid_args", "input missing", null); return }
                    // convert to float array with shape [1,200,99]
                    val flat = FloatArray(inputList.size)
                    for (i in inputList.indices) flat[i] = inputList[i].toFloat()

                    // Prepare input/output buffers via reflection
                    val interpreterClass = Class.forName("org.tensorflow.lite.Interpreter")
                    val runMethod = interpreterClass.getMethod("run", Any::class.java, Any::class.java)

                    // Input shape [1,200,99]
                    val inShape = arrayOf(arrayOf(Array(200) { FloatArray(99) })) // heavy but placeholder
                    // Fill inShape with flat values
                    var idx = 0
                    for (i in 0 until 1) {
                        for (j in 0 until 200) {
                            for (k in 0 until 99) {
                                (inShape[i][j] as FloatArray)[k] = flat[idx++]
                            }
                        }
                    }

                    val outShape = Array(1) { FloatArray(24) }
                    runMethod.invoke(interpreter, inShape, outShape)

                    // Convert output to List<Double>
                    val outList = mutableListOf<Double>()
                    for (d in outShape[0]) outList.add(d.toDouble())
                    result.success(outList)
                } catch (e: Exception) {
                    Log.e(TAG, "runInference failed: ${e.message}")
                    result.error("run_error", "runInference failed: ${e.message}", null)
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun copyAssetToFile(assetPath: String): File {
        val afd: AssetFileDescriptor = context.assets.openFd(assetPath)
        val input = afd.createInputStream()
        val outFile = File(context.cacheDir, "model_${System.currentTimeMillis()}.tflite")
        val out = FileOutputStream(outFile)
        val buffer = ByteArray(4 * 1024)
        var read: Int
        while (input.read(buffer).also { read = it } != -1) {
            out.write(buffer, 0, read)
        }
        out.flush()
        out.close()
        input.close()
        return outFile
    }
}
