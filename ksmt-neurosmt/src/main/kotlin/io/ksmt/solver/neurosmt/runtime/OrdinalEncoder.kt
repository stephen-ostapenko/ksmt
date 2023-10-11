package io.ksmt.solver.neurosmt.runtime

import java.io.InputStream

const val UNKNOWN_VALUE = 1999

// wrapper for single-feature sklearn OrdinalEncoder (for each string we should provide its ordinal)
// used to convert strings to integers
class OrdinalEncoder(ordinalsAsStream: InputStream, private val unknownValue: Int = UNKNOWN_VALUE) {
    private val lookup = HashMap<String, Int>()

    init {
        ordinalsAsStream.readAllBytes().decodeToString().lines().forEachIndexed { index, s ->
            lookup[s] = index
        }
    }

    fun getOrdinal(s: String): Int {
        return lookup[s] ?: unknownValue
    }
}