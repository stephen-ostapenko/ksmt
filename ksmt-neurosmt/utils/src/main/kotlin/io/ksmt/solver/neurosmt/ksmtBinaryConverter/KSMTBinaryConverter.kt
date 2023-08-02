package io.ksmt.solver.neurosmt.ksmtBinaryConverter

import io.ksmt.KContext
import io.ksmt.parser.KSMTLibParseException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.FormulaGraphExtractor
import io.ksmt.solver.neurosmt.deserialize
import io.ksmt.solver.neurosmt.getAnswerForTest
import io.ksmt.solver.z3.KZ3SMTLibParser
import me.tongfei.progressbar.ProgressBar
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name
import kotlin.time.Duration.Companion.seconds

fun main(args: Array<String>) {
    val inputRoot = args[0]
    val outputRoot = args[1]
    val timeout = args[2].toInt().seconds

    val files = Files.walk(Path.of(inputRoot)).filter { it.isRegularFile() }

    var sat = 0; var unsat = 0; var skipped = 0

    val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

    var curIdx = 0
    ProgressBar.wrap(files, "converting ksmt binary files").forEach {
        val assertList = try {
            deserialize(ctx, FileInputStream(it.toFile()))
        } catch (e: Exception) {
            skipped++
            return@forEach
        }

        val answer = getAnswerForTest(ctx, assertList, timeout)

        if (answer == KSolverStatus.UNKNOWN) {
            skipped++
            return@forEach
        }

        with(ctx) {
            val formula = when (assertList.size) {
                0 -> {
                    skipped++
                    return@forEach
                }
                1 -> {
                    assertList[0]
                }
                else -> {
                    mkAnd(assertList)
                }
            }

            val outputStream = FileOutputStream("$outputRoot/$curIdx-${answer.toString().lowercase()}")
            outputStream.write("; $it\n".encodeToByteArray())

            val extractor = FormulaGraphExtractor(ctx, formula, outputStream)
            extractor.extractGraph()
        }

        when (answer) {
            KSolverStatus.SAT -> sat++
            KSolverStatus.UNSAT -> unsat++
            else -> { /* can't happen */ }
        }

        curIdx++
    }

    println()
    println("sat: $sat; unsat: $unsat; skipped: $skipped")
}