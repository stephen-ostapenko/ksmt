package io.ksmt.solver.neurosmt.smt2Converter

import io.ksmt.KContext
import io.ksmt.parser.KSMTLibParseException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.FormulaGraphExtractor
import io.ksmt.solver.neurosmt.checkFormulaCategory
import io.ksmt.solver.neurosmt.getAnswerForTest
import io.ksmt.solver.z3.KZ3SMTLibParser
import me.tongfei.progressbar.ProgressBar
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name

// tool to convert formulas from .smt2 format to their structure graphs
fun main(args: Array<String>) {
    val inputRoot = args[0]
    val outputRoot = args[1]

    if (args[2] !in listOf("graph", "raw")) {
        throw IllegalArgumentException("specify output format (raw/graph)")
    }
    val graphOutput = args[2] == "graph"
    val simplify = args.getOrNull(3) == "simplify"

    val files = Files.walk(Path.of(inputRoot)).filter { it.isRegularFile() }

    var ok = 0; var fail = 0
    var sat = 0; var unsat = 0; var skipped = 0

    val ctx = KContext(
        simplificationMode = if (simplify) {
            KContext.SimplificationMode.SIMPLIFY
        } else {
            KContext.SimplificationMode.NO_SIMPLIFY
        },
        astManagementMode = KContext.AstManagementMode.GC
    )

    var curIdx = 0
    ProgressBar.wrap(files.toList(), "converting smt2 files").forEach {
        if (!it.name.endsWith(".smt2")) {
            return@forEach
        }

        if (!checkFormulaCategory(it, "industrial")) {
            skipped++
            return@forEach
        }

        with(ctx) {
            val assertList = try {
                KZ3SMTLibParser(ctx).parse(it)
            } catch (e: KSMTLibParseException) {
                fail++
                println("e: error in $it: $e")
                return@forEach
            } catch (e: NotImplementedError) {
                fail++
                println("e: error in $it: $e")
                return@forEach
            }

            val formula = when (assertList.size) {
                0 -> {
                    skipped++
                    return@forEach
                }
                1 -> {
                    ok++
                    assertList[0]
                }
                else -> {
                    ok++
                    mkAnd(assertList)
                }
            }

            val answer = getAnswerForTest(it)

            if (answer == KSolverStatus.UNKNOWN) {
                skipped++
                return@forEach
            }

            var relFile = it.toFile().relativeTo(File(inputRoot))
            while (relFile.parentFile != null && relFile.parentFile.parentFile != null) {
                relFile = relFile.parentFile
            }
            val outputDir = File(outputRoot, relFile.toString().replace('/', '-'))
            outputDir.mkdirs()

            val outputFile = File("$outputDir/$curIdx-${answer.toString().lowercase()}")
            val outputStream = FileOutputStream(outputFile)

            if (graphOutput) {
                outputStream.write("; $it\n".encodeToByteArray())
                val extractor = FormulaGraphExtractor(ctx, formula, outputStream)
                extractor.extractGraph()
            } else {
                val formulaString = StringBuilder()
                formula.print(formulaString)
                outputStream.write(formulaString.toString().toByteArray())
            }

            when (answer) {
                KSolverStatus.SAT -> sat++
                KSolverStatus.UNSAT -> unsat++
                else -> { /* can't happen */ }
            }
        }

        curIdx++
    }

    println()
    println("processed: $ok; failed: $fail")
    println("sat: $sat; unsat: $unsat; skipped: $skipped")
}