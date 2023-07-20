package io.ksmt.solver.neurosmt.smt2converter

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import java.io.OutputStream
import java.util.IdentityHashMap

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

        //println(expr::class)
        //println("${expr.decl.name} ${expr.args.size}")
        //outputStream.writer().write("${expr.decl.name} ${expr.args.size}\n")
        //outputStream.writer().flush()
        /*
        println(expr.args)
        println("${expr.decl} ${expr.decl.name} ${expr.decl.argSorts} ${expr.sort}")
        for (child in expr.args) {
            println((child as KApp<*, *>).decl.name)
        }
        */

        return expr
    }

    fun <T : KSort> writeSymbolicVariable(symbol: KConst<T>) {
        writer.write("SYMBOLIC; ${symbol.sort}\n")
    }

    fun <T : KSort> writeValue(value: KInterpretedValue<T>) {
        writer.write("VALUE; ${value.decl.sort}\n")
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