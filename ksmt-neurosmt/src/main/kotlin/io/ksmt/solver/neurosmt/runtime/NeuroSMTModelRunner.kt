package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OrtEnvironment
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KTrue
import java.io.InputStream
import kotlin.math.exp

// wrapper for NeuroSMT model
// runs the whole model pipeline
class NeuroSMTModelRunner(
    val ctx: KContext,
    ordinalsAsStream: InputStream,
    embeddingLayerAsStream: InputStream,
    convLayerAsStream: InputStream,
    decoderAsStream: InputStream
) {
    val env = OrtEnvironment.getEnvironment()

    val ordinalEncoder = OrdinalEncoder(ordinalsAsStream)
    val embeddingLayer = ONNXModel(env, embeddingLayerAsStream)
    val convLayer = ONNXModel(env, convLayerAsStream)

    val decoder = ONNXModel(env, decoderAsStream)

    fun run(expr: KExpr<*>): Float {
        if (expr is KTrue) {
            return 1f
        }
        if (expr is KFalse) {
            return 0f
        }

        val encoder = ExprEncoder(ctx, env, ordinalEncoder, embeddingLayer, convLayer)
        val exprFeatures = encoder.encodeExpr(expr)
        val result = decoder.forward(mapOf("expr_features" to exprFeatures))
        val logit = result.floatBuffer[0]

        return 1f / (1f + exp(-logit)) // sigmoid calculation
    }

    fun close() {
        embeddingLayer.close()
        convLayer.close()
        decoder.close()
    }
}