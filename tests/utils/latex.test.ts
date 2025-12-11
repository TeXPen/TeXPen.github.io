import { describe, expect, test } from "vitest";
import { removeStyle, addNewlines } from "../../utils/latex";

describe("latexUtils", () => {
  describe("removeStyle", () => {
    test("removes \\bm", () => {
      expect(removeStyle("\\bm{x}")).toBe("x");
    });

    test("removes \\boldsymbol", () => {
      expect(removeStyle("\\boldsymbol{y}")).toBe("y");
    });

    test("removes \\textit", () => {
      expect(removeStyle("\\textit{text}")).toBe("text");
    });

    test("removes \\textbf", () => {
      expect(removeStyle("\\textbf{bold}")).toBe("bold");
    });

    test("removes \\mathbf", () => {
      expect(removeStyle("\\mathbf{B}")).toBe("B");
    });

    test("handles nested", () => {
      expect(removeStyle("\\bm{\\textit{x}}")).toBe("x");
    });

    test("handles complex expressions", () => {
      // The implementation inserts spaces during replacement: \bm{b} -> " b "
      expect(removeStyle("a + \\bm{b} = c")).toBe("a +  b  = c");
    });
  });

  describe("addNewlines", () => {
    test("adds newlines around begin/end", () => {
      const input = "text \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix} text";
      const output = addNewlines(input);
      expect(output).toContain("\n\\begin{bmatrix}\n");
      expect(output).toContain("\n\\end{bmatrix}\n");
    });

    test("adds newlines after double backslash", () => {
      const input = "row 1 \\\\ row 2";
      const output = addNewlines(input);
      // The space after \\ is consumed and replaced by \n
      expect(output).toBe("row 1 \\\\\nrow 2");
    });

    test("collapses multiple newlines", () => {
      const input = "line1\n\n\nline2";
      expect(addNewlines(input)).toBe("line1\nline2");
    });
  });
});
