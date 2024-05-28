package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.*
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.*

// expression encoder
// walks on an expression and calculates state for each node
// based on states for its children (which are already calculated at that moment)
class ExprEncoder(
    override val ctx: KContext,
    val env: OrtEnvironment,
    val ordinalEncoder: OrdinalEncoder,
    val embeddingLayer: ONNXModel,
    val convLayer: ONNXModel
) : KNonRecursiveTransformer(ctx) {

    private val exprToState = IdentityHashMap<KExpr<*>, OnnxTensor>()

    fun encodeExpr(expr: KExpr<*>): OnnxTensor {
        apply(expr)

        return exprToState[expr] ?: error("expression state wasn't calculated yet [$expr]")
    }

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        val state = when (expr) {
            is KConst<*> -> calcSymbolicVariableState(expr)
            is KInterpretedValue<*> -> calcValueState(expr)
            else -> calcAppState(expr)
        }

        exprToState[expr] = state

        return expr
    }

    private fun getNodeEmbedding(key: String): OnnxTensor {
        val nodeLabel = ordinalEncoder.getOrdinal(key)
        val labelTensor = OnnxTensor.createTensor(
            env, IntBuffer.allocate(1).put(nodeLabel).rewind(), longArrayOf(1, 1)
        )

        return embeddingLayer.forward(mapOf("node_labels" to labelTensor))
    }

    private fun createEdgeTensor(childrenCnt: Int): OnnxTensor {
        val edges = listOf(
            List(childrenCnt) { it + 1L },
            List(childrenCnt) { 0L }
        )

        val buffer = LongBuffer.allocate(childrenCnt * 2)
        edges.forEach { row ->
            row.forEach { node ->
                buffer.put(node)
            }
        }
        buffer.rewind()

        return OnnxTensor.createTensor(env, buffer, longArrayOf(2, childrenCnt.toLong()))
    }

    private fun <T : KSort, A : KSort> calcAppState(expr: KApp<T, A>): OnnxTensor {
        val childrenStates = expr.args.map {
            when (it) {
                is KArrayLambda<*, *> -> calcLambdaState(it)
                else -> exprToState[it] ?: error("expression state wasn't calculated yet [$it]")
            }
        }
        val childrenCnt = childrenStates.size

        val nodeEmbedding = getNodeEmbedding(expr.decl.name)
        val embeddingSize = nodeEmbedding.info.shape.reduce { acc, l -> acc * l }

        val buffer = FloatBuffer.allocate((1 + childrenCnt) * embeddingSize.toInt())
        buffer.put(nodeEmbedding.floatBuffer)
        childrenStates.forEach {
            buffer.put(it.floatBuffer)
        }
        buffer.rewind()

        val nodeFeatures = OnnxTensor.createTensor(env, buffer, longArrayOf(1L + childrenCnt, embeddingSize))
        val edges = createEdgeTensor(childrenStates.size)
        val result = convLayer.forward(mapOf("node_features" to nodeFeatures, "edges" to edges))

        return OnnxTensor.createTensor(env, result.floatBuffer.slice(0, embeddingSize.toInt()), longArrayOf(1L, embeddingSize))
    }

    private fun <T : KSort> getSortStringRepresentation(sort: T): String {
        return when (sort) {
            is KBoolSort -> "Bool"
            is KIntSort -> "Int"
            is KRealSort -> "Real"
            is KBvSort -> "BitVec"
            is KFpSort -> "FP"
            is KFpRoundingModeSort -> "FP_RM"
            is KArraySortBase<*> -> "Array<${sort.domainSorts.joinToString(",")}>"
            is KUninterpretedSort -> sort.name
            else -> error("unknown symbolic sort: ${sort::class.simpleName}")
        }
    }

    private fun <T : KSort> calcSymbolicVariableState(symbol: KConst<T>): OnnxTensor {
        val key = "SYMBOLIC;" + getSortStringRepresentation(symbol.decl.sort)
        return getNodeEmbedding(key)
    }

    private fun calcLambdaState(lambda: KArrayLambda<*, *>): OnnxTensor {
        val key = "SYMBOLIC;" + getSortStringRepresentation(lambda.sort)
        return getNodeEmbedding(key)
    }

    private fun <T : KSort> calcValueState(value: KInterpretedValue<T>): OnnxTensor {
        val key = "VALUE;" + getSortStringRepresentation(value.decl.sort)
        return getNodeEmbedding(key)
    }
}