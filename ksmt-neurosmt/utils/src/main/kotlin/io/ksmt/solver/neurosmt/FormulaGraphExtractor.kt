package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.*
import java.io.OutputStream
import java.util.*

class FormulaGraphExtractor(
    override val ctx: KContext,
    val formula: KExpr<KBoolSort>,
    outputStream: OutputStream
) : KNonRecursiveTransformer(ctx) {

    private val exprToVertexID = IdentityHashMap<KExpr<*>, Long>()
    private var currentID = 0L

    private val writer = outputStream.bufferedWriter()

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        exprToVertexID[expr] = currentID++

        when (expr) {
            is KInterpretedValue<*> -> writeValue(expr)
            is KConst<*> -> writeSymbolicVariable(expr)
            else -> writeApp(expr)
        }

        return expr
    }

    fun <T : KSort> writeSymbolicVariable(symbol: KConst<T>) {
        when (symbol.decl.sort) {
            is KBoolSort -> writer.write("SYMBOLIC; Bool\n")
            is KBvSort -> writer.write("SYMBOLIC; BitVec\n")
            is KFpSort -> writer.write("SYMBOLIC; FP\n")
            is KFpRoundingModeSort -> writer.write("SYMBOLIC; FP_RM\n")
            is KArraySortBase<*> -> writer.write("SYMBOLIC; Array\n")
            is KUninterpretedSort -> writer.write("SYMBOLIC; Unint\n")
            else -> error("unknown symbolic sort: ${symbol.sort::class.simpleName}")
        }
    }

    fun <T : KSort> writeValue(value: KInterpretedValue<T>) {
        when (value.decl.sort) {
            is KBoolSort -> writer.write("VALUE; Bool\n")
            is KBvSort -> writer.write("VALUE; BitVec\n")
            is KFpSort -> writer.write("VALUE; FP\n")
            is KFpRoundingModeSort -> writer.write("VALUE; FP_RM\n")
            is KArraySortBase<*> -> writer.write("VALUE; Array\n")
            is KUninterpretedSort -> writer.write("VALUE; Unint\n")
            else -> error("unknown value sort: ${value.decl.sort::class.simpleName}")
        }
    }

    fun <T : KSort, A : KSort> writeApp(expr: KApp<T, A>) {
        writer.write("${expr.decl.name};")
        for (child in expr.args) {
            writer.write(" ${exprToVertexID[child]}")
        }
        writer.newLine()
    }

    fun extractGraph() {
        apply(formula)
        writer.close()
    }
}