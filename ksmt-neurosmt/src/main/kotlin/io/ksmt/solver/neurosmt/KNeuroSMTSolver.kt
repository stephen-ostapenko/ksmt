package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.runtime.NeuroSMTModelRunner
import io.ksmt.sort.KBoolSort
import kotlin.math.min
import kotlin.time.Duration

class KNeuroSMTSolver(
    private val ctx: KContext,
    private val threshold: Double = 0.5
) : KSolver<KNeuroSMTSolverConfiguration> {

    companion object {
        private const val ENCODER_ORDINALS_RESOURCE_FILE_PATH = "encoder-ordinals"
        private const val EMBEDDINGS_RESOURCE_FILE_PATH = "embeddings.onnx"
        private const val CONV_RESOURCE_FILE_PATH = "conv.onnx"
        private const val DECODER_RESOURCE_FILE_PATH = "decoder.onnx"
    }

    private val classLoader = this.javaClass.classLoader
    private val encoderOrdinalsResourceAsStream = classLoader.getResourceAsStream(ENCODER_ORDINALS_RESOURCE_FILE_PATH)
        ?: error("encoder ordinals are not found")
    private val embeddingLayerResourceAsStream = classLoader.getResourceAsStream(EMBEDDINGS_RESOURCE_FILE_PATH)
        ?: error("embedding layer resources are not found")
    private val convLayerResourceAsStream = classLoader.getResourceAsStream(CONV_RESOURCE_FILE_PATH)
        ?: error("conv layer resources are not found")
    private val decoderResourceAsStream = classLoader.getResourceAsStream(DECODER_RESOURCE_FILE_PATH)
        ?: error("decoder resources are not found")

    private val modelRunner = NeuroSMTModelRunner(
        ctx,
        encoderOrdinalsResourceAsStream,
        embeddingLayerResourceAsStream,
        convLayerResourceAsStream,
        decoderResourceAsStream
    )

    private val asserts = mutableListOf<MutableList<KExpr<KBoolSort>>>(mutableListOf())

    override fun configure(configurator: KNeuroSMTSolverConfiguration.() -> Unit) {
        TODO("Not yet implemented")
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        asserts.last().add(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        TODO("Not yet implemented")
    }

    override fun push() {
        asserts.add(mutableListOf())
    }

    override fun pop(n: UInt) {
        repeat(min(n.toInt(), asserts.size)) {
            asserts.removeLast()
        }
    }

    override fun check(timeout: Duration): KSolverStatus {
        val prob = with(ctx) {
            modelRunner.run(mkAnd(asserts.flatten()))
        }

        return if (prob > threshold) {
            KSolverStatus.SAT
        } else {
            KSolverStatus.UNSAT
        }
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        val prob = with(ctx) {
            modelRunner.run(mkAnd(asserts.flatten() + assumptions))
        }

        return if (prob > threshold) {
            KSolverStatus.SAT
        } else {
            KSolverStatus.UNSAT
        }
    }

    override fun model(): KModel {
        TODO("Not yet implemented")
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        TODO("Not yet implemented")
    }

    override fun reasonOfUnknown(): String {
        TODO("Not yet implemented")
    }

    override fun interrupt() {
        TODO("Not yet implemented")
    }

    override fun close() {
        modelRunner.close()
        asserts.clear()
    }
}