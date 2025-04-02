/* SUGGESTED CHANGES FROM DEBUG ASSISTANT */

/* SUGGESTION 1 */
            // Calculate the absolute index (0-based) of the middle sample in the original sequence
            // Index i corresponds to the *end* of the current window.
            // The window covers indices [i - WINDOW_LENGTH + 1, i].
            // The middle element is at absolute index: (i - WINDOW_LENGTH + 1) + MIDDLE_LOCATION
            Index_t candidateIndexAbs = i - WINDOW_LENGTH + 1 + MIDDLE_LOCATION;


/* SUGGESTION 2 */
            // Calculate the absolute index (0-based) of the middle sample in the original sequence
            // Index i corresponds to the *end* of the current window.
            // The window covers indices [i - WINDOW_LENGTH + 1, i].
            // The middle element (at buffer index MIDDLE_LOCATION) corresponds to the sample
            // read at iteration i - MIDDLE_LOCATION.
            // Based on simulation results showing a +1 offset vs reference, adjust the index:
            Index_t candidateIndexAbs = i - MIDDLE_LOCATION - 1;

            // --- Alternatively, using the original formula structure: ---
            // Index_t candidateIndexAbs = i - WINDOW_LENGTH + MIDDLE_LOCATION; // Equivalent to i - MIDDLE_LOCATION - 1


