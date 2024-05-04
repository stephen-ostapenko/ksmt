package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.*
import java.io.OutputStream
import java.util.*

// serializer of ksmt formula to simple format of formula's structure graph
class FormulaGraphExtractor(
    override val ctx: KContext,
    val formula: KExpr<KBoolSort>,
    outputStream: OutputStream
) : KNonRecursiveTransformer(ctx) {

    private val exprToVertexID = IdentityHashMap<KExpr<*>, Long>()
    private var currentID = 0L

    private val writer = outputStream.bufferedWriter()

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        when (expr) {
            is KInterpretedValue<*> -> writeValue(expr)
            is KConst<*> -> writeSymbolicVariable(expr)
            else -> writeApp(expr)
        }
        writer.newLine()

        exprToVertexID[expr] = currentID++

        return expr
    }

    private fun <T : KSort> writeSymbolicVariable(symbol: KConst<T>) {
        writer.write("SYMBOLIC;")
        writeSort(symbol.decl.sort)
    }

    private fun <T : KSort> writeValue(value: KInterpretedValue<T>) {
        writer.write("VALUE;")
        writeSort(value.decl.sort)
    }

    private fun <T : KSort> writeSort(sort: T) {
        when (sort) {
            is KBoolSort -> writer.write("Bool")
            is KIntSort -> writer.write("Int")
            is KRealSort -> writer.write("Real")
            is KBvSort -> writer.write("BitVec")
            is KFpSort -> writer.write("FP")
            is KFpRoundingModeSort -> writer.write("FP_RM")
            is KArraySortBase<*> -> writer.write("Array<${sort.domainSorts.joinToString(",")}>")
            is KUninterpretedSort -> writer.write(sort.name)
            else -> error("unknown sort: ${sort::class.simpleName}")
        }
    }

    private fun <D : KSort, R : KSort> writeLambda(expr: KArrayLambda<D, R>) {
        if (!exprToVertexID.containsKey(expr)) {
            exprToVertexID[expr] = currentID++
        }

        writer.write("SYMBOLIC;")
        writeSort(expr.sort)
        writer.newLine()
    }

    private fun <T : KSort, A : KSort> writeApp(expr: KApp<T, A>) {
        for (child in expr.args) {
            if (child is KArrayLambda<*, *>) {
                writeLambda(child)
            }
        }

        writer.write("${expr.decl.name};")

        for (child in expr.args) {
            writer.write(" ${exprToVertexID[child]}")
        }
    }

    fun extractGraph() {
        apply(formula)
        writer.close()
    }
}