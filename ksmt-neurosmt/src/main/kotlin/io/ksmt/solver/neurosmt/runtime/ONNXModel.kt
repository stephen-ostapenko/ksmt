package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.io.InputStream

// wrapper for any exported ONNX model
class ONNXModel(env: OrtEnvironment, modelFileAsStream: InputStream) : AutoCloseable {
    val session = env.createSession(modelFileAsStream.readAllBytes())

    fun forward(input: Map<String, OnnxTensor>): OnnxTensor {
        val result = session.run(input)
        return result.get(0) as OnnxTensor
    }

    override fun close() {
        session.close()
    }
}