
/**
 * Ported from TexTeller/texteller/utils/latex.py
 */

function change(
  inputStr: string,
  oldInst: string,
  newInst: string,
  oldSurrL: string,
  oldSurrR: string,
  newSurrL: string,
  newSurrR: string
): string {
  let result = '';
  let i = 0;
  const n = inputStr.length;

  while (i < n) {
    if (inputStr.substring(i, i + oldInst.length) === oldInst) {
      // check if the oldInst is followed by oldSurrL
      const start = i + oldInst.length;

      if (start < n && inputStr[start] === oldSurrL) {
        // found an oldInst followed by oldSurrL, now look for the matching oldSurrR
        let count = 1;
        let j = start + 1;
        let escaped = false;

        while (j < n && count > 0) {
          if (inputStr[j] === '\\' && !escaped) {
            escaped = true;
            j++;
            continue;
          }
          if (inputStr[j] === oldSurrR && !escaped) {
            count--;
            if (count === 0) {
              break;
            }
          } else if (inputStr[j] === oldSurrL && !escaped) {
            count++;
          }
          escaped = false;
          j++;
        }

        if (count === 0) {
          // Found matching closing brace
          const innerContent = inputStr.substring(start + 1, j);
          // Replace the content with new pattern
          result += newInst + newSurrL + innerContent + newSurrR;
          i = j + 1;
          continue;
        } else {
          // Unbalanced
          console.warn('Warning: unbalanced surrogate pair in input string');
          result += newInst + newSurrL;
          i = start + 1;
          continue;
        }
      } else {
        // Not followed by oldSurrL, just append oldInst
        result += inputStr.substring(i, start);
        i = start;
        continue; // Continue loop from new i
      }
    } else {
      result += inputStr[i];
      i++;
    }
  }

  if (oldInst !== newInst && result.includes(oldInst + oldSurrL)) {
    return change(result, oldInst, newInst, oldSurrL, oldSurrR, newSurrL, newSurrR);
  } else {
    return result;
  }
}

function findSubstringPositions(str: string, substring: string): number[] {
  const positions: number[] = [];
  let pos = str.indexOf(substring);
  while (pos !== -1) {
    positions.push(pos);
    pos = str.indexOf(substring, pos + 1);
  }
  return positions;
}

// In the Python version, `change_all` iterates through positions in reverse.
// However, since `change` (the recursive function) processes the whole string and handles recursion,
// we might just need to call `change` directly.
// The Python `change_all` finds all occurrences of `old_inst + old_surr_l` and calls `_change` on the substring starting from there?
// Wait, the Python `change_all` logic is:
// pos = _find_substring_positions(input_str, old_inst + old_surr_l)
// for p in pos[::-1]:
//    res[p:] = list(_change("".join(res[p:]), ...))
// This seems to apply `_change` to the suffix of the string starting at each match, going backwards.
// But `_change` itself iterates through the string.
// Actually, `_change` in Python iterates from 0 to n.
// If we just call `change` on the whole string, it should handle all occurrences.
// The Python `change_all` might be redundant or doing something specific with overlapping/nested things that `_change` alone doesn't catch if it modifies the string?
// But `_change` is recursive: `if ... in result: return _change(...)`.
// So `change` on the full string should be sufficient and equivalent to `change_all` if `change` handles all instances.
// Let's stick to calling `change` on the full string.

export function removeStyle(inputStr: string): string {
  let s = inputStr;
  s = change(s, '\\bm', ' ', '{', '}', '', ' ');
  s = change(s, '\\boldsymbol', ' ', '{', '}', '', ' ');
  s = change(s, '\\textit', ' ', '{', '}', '', ' ');
  s = change(s, '\\textbf', ' ', '{', '}', '', ' ');
  s = change(s, '\\mathbf', ' ', '{', '}', '', ' ');
  return s.trim();
}

export function addNewlines(latexStr: string): string {
  let processedStr = latexStr;

  // 1. Replace whitespace around \begin{...} with \n...\n
  processedStr = processedStr.replace(/\s*(\\begin\{[^}]*\})\s*/g, '\n$1\n');

  // 2. Replace whitespace around \end{...} with \n...\n
  processedStr = processedStr.replace(/\s*(\\end\{[^}]*\})\s*/g, '\n$1\n');

  // 3. Add newline after \\ (if not already followed by newline)
  // JS regex lookahead: x(?=y) matches x only if x is followed by y
  // Negative lookahead: x(?!y) matches x only if x is not followed by y
  processedStr = processedStr.replace(/\\\\(?!\n| )|\\\\ /g, '\\\\\n');

  // 4. Cleanup: Collapse multiple consecutive newlines into a single newline.
  processedStr = processedStr.replace(/\n{2,}/g, '\n');

  return processedStr.trim();
}
