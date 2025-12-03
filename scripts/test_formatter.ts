
import { formatLatex } from '../services/latexFormatter';

const testCases = [
  {
    name: "Basic Text",
    input: "Hello world",
    expected: "Hello world"
  },
  {
    name: "Itemize Environment",
    input: "\\begin{itemize}\n\\item Item 1\n\\item Item 2\n\\end{itemize}",
    expected: "\\begin{itemize}\n    \\item Item 1\n    \\item Item 2\n\\end{itemize}"
  },
  {
    name: "Nested Environments",
    input: "\\begin{document}\n\\begin{section}\nText\n\\end{section}\n\\end{document}",
    expected: "\\begin{document}\n    \\begin{section}\n        Text\n    \\end{section}\n\\end{document}"
  },
  {
    name: "Wrapping",
    input: "This is a very long line that should be wrapped because it exceeds the default wrap length of 80 characters. It really should be wrapped.",
    // Note: Default wrap is false in DEFAULT_ARGS, so it shouldn't wrap unless we enable it.
    // But let's check if it preserves it.
    expected: "This is a very long line that should be wrapped because it exceeds the default wrap length of 80 characters. It really should be wrapped."
  }
];

console.log("Running LaTeX Formatter Tests...");

let passed = 0;
let failed = 0;

for (const test of testCases) {
  try {
    const result = formatLatex(test.input);
    if (result === test.expected) {
      console.log(`[PASS] ${test.name}`);
      passed++;
    } else {
      console.error(`[FAIL] ${test.name}`);
      console.error(`  Expected:\n${JSON.stringify(test.expected)}`);
      console.error(`  Actual:\n${JSON.stringify(result)}`);
      failed++;
    }
  } catch (e) {
    console.error(`[ERROR] ${test.name}: ${e}`);
    failed++;
  }
}

console.log(`\nSummary: ${passed} passed, ${failed} failed.`);

if (failed > 0) {
  process.exit(1);
}
