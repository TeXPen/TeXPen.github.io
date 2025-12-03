
/**
 * TypeScript implementation of tex-fmt, a LaTeX formatter.
 * Ported from TexTeller's format.py
 */

// Constants
const LINE_END = "\n";
const ITEM = "\\item";
const DOC_BEGIN = "\\begin{document}";
const DOC_END = "\\end{document}";
const ENV_BEGIN = "\\begin{";
const ENV_END = "\\end{";
const TEXT_LINE_START = "";
const COMMENT_LINE_START = "% ";

// Opening and closing delimiters
const OPENS = ["{", "(", "["];
const CLOSES = ["}", ")", "]"];

// Names of LaTeX verbatim environments
const VERBATIMS = ["verbatim", "Verbatim", "lstlisting", "minted", "comment"];
const VERBATIMS_BEGIN = VERBATIMS.map(v => `\\begin{${v}}`);
const VERBATIMS_END = VERBATIMS.map(v => `\\end{${v}}`);

// Regex patterns for sectioning commands
const SPLITTING = [
  "\\\\begin\\{",
  "\\\\end\\{",
  "\\\\item(?:$|[^a-zA-Z])",
  "\\\\(?:sub){0,2}section\\*?\\{",
  "\\\\chapter\\*?\\{",
  "\\\\part\\*?\\{",
];

// Compiled regexes
const SPLITTING_STRING = `(${SPLITTING.join('|')})`;
const RE_SPLITTING = new RegExp(SPLITTING_STRING);
const RE_SPLITTING_SHARED_LINE = new RegExp(`.*${SPLITTING_STRING}.*`);
const RE_SPLITTING_SHARED_LINE_CAPTURE = new RegExp(`(?<prev>.*?)(?<env>${SPLITTING_STRING}.*)`);

const RE_NEWLINES = new RegExp(`${LINE_END}${LINE_END}(${LINE_END})+`, 'g');
const RE_TRAIL = new RegExp(` +${LINE_END}`, 'g'); // Note: JS RegExp for multiline might need flags

export interface Args {
  tabchar: string;
  tabsize: number;
  wrap: boolean;
  wraplen: number;
  wrapmin: number;
  lists: string[];
  verbosity: number;
}

export const DEFAULT_ARGS: Args = {
  tabchar: " ",
  tabsize: 4,
  wrap: false,
  wraplen: 80,
  wrapmin: 40,
  lists: [],
  verbosity: 0
};

export class Ignore {
  constructor(public actual: boolean = false, public visual: boolean = false) { }
  static new() { return new Ignore(false, false); }
}

export class Verbatim {
  constructor(public actual: number = 0, public visual: boolean = false) { }
  static new() { return new Verbatim(0, false); }
}

export class Indent {
  constructor(public actual: number = 0, public visual: number = 0) { }
  static new() { return new Indent(0, 0); }
}

export class State {
  linum_old: number = 1;
  linum_new: number = 1;
  ignore: Ignore;
  indent: Indent;
  verbatim: Verbatim;
  linum_last_zero_indent: number = 1;

  constructor(init?: Partial<State>) {
    this.linum_old = init?.linum_old ?? 1;
    this.linum_new = init?.linum_new ?? 1;
    this.ignore = init?.ignore ?? Ignore.new();
    this.indent = init?.indent ?? Indent.new();
    this.verbatim = init?.verbatim ?? Verbatim.new();
    this.linum_last_zero_indent = init?.linum_last_zero_indent ?? 1;
  }
}

export class Pattern {
  constructor(
    public contains_env_begin: boolean = false,
    public contains_env_end: boolean = false,
    public contains_item: boolean = false,
    public contains_splitting: boolean = false,
    public contains_comment: boolean = false
  ) { }

  static new(s: string): Pattern {
    if (RE_SPLITTING.test(s)) {
      return new Pattern(
        s.includes(ENV_BEGIN),
        s.includes(ENV_END),
        s.includes(ITEM),
        true,
        s.includes("%")
      );
    } else {
      return new Pattern(
        false,
        false,
        false,
        false,
        s.includes("%")
      );
    }
  }
}

export interface Log {
  level: string;
  file: string;
  message: string;
  linum_new?: number;
  linum_old?: number;
  line?: string;
}

// Helper Functions

function findCommentIndex(line: string, pattern: Pattern): number | null {
  if (!pattern.contains_comment) return null;

  let inCommand = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === "\\") {
      inCommand = true;
    } else if (inCommand && !/[a-zA-Z]/.test(c)) {
      inCommand = false;
    } else if (c === "%" && !inCommand) {
      return i;
    }
  }
  return null;
}

function containsIgnoreSkip(line: string): boolean {
  return line.endsWith("% tex-fmt: skip");
}

function containsIgnoreBegin(line: string): boolean {
  return line.endsWith("% tex-fmt: off");
}

function containsIgnoreEnd(line: string): boolean {
  return line.endsWith("% tex-fmt: on");
}

function getIgnore(line: string, state: State, logs: Log[], file: string, warn: boolean): Ignore {
  const skip = containsIgnoreSkip(line);
  const begin = containsIgnoreBegin(line);
  const end = containsIgnoreEnd(line);

  let actual: boolean;
  let visual: boolean;

  if (skip) {
    actual = state.ignore.actual;
    visual = true;
  } else if (begin) {
    actual = true;
    visual = true;
    if (warn && state.ignore.actual) {
      logs.push({
        level: "WARN",
        file,
        message: "Cannot begin ignore block:",
        linum_new: state.linum_new,
        linum_old: state.linum_old,
        line
      });
    }
  } else if (end) {
    actual = false;
    visual = true;
    if (warn && !state.ignore.actual) {
      logs.push({
        level: "WARN",
        file,
        message: "No ignore block to end.",
        linum_new: state.linum_new,
        linum_old: state.linum_old,
        line
      });
    }
  } else {
    actual = state.ignore.actual;
    visual = state.ignore.actual;
  }

  return new Ignore(actual, visual);
}

function getVerbatimDiff(line: string, pattern: Pattern): number {
  if (pattern.contains_env_begin && VERBATIMS_BEGIN.some(r => line.includes(r))) {
    return 1;
  } else if (pattern.contains_env_end && VERBATIMS_END.some(r => line.includes(r))) {
    return -1;
  } else {
    return 0;
  }
}

function getVerbatim(line: string, state: State, logs: Log[], file: string, warn: boolean, pattern: Pattern): Verbatim {
  const diff = getVerbatimDiff(line, pattern);
  const actual = state.verbatim.actual + diff;
  const visual = actual > 0 || state.verbatim.actual > 0;

  if (warn && actual < 0) {
    logs.push({
      level: "WARN",
      file,
      message: "Verbatim count is negative.",
      linum_new: state.linum_new,
      linum_old: state.linum_old,
      line
    });
  }

  return new Verbatim(actual, visual);
}

function getDiff(line: string, pattern: Pattern, listsBegin: string[], listsEnd: string[]): number {
  let diff = 0;

  // Other environments get single indents
  if (pattern.contains_env_begin && line.includes(ENV_BEGIN)) {
    // Documents get no global indentation
    if (line.includes(DOC_BEGIN)) {
      return 0;
    }
    diff += 1;
    diff += listsBegin.some(r => line.includes(r)) ? 1 : 0;
  } else if (pattern.contains_env_end && line.includes(ENV_END)) {
    // Documents get no global indentation
    if (line.includes(DOC_END)) {
      return 0;
    }
    diff -= 1;
    diff -= listsEnd.some(r => line.includes(r)) ? 1 : 0;
  }

  // Indent for delimiters
  for (const c of line) {
    if (OPENS.includes(c)) {
      diff += 1;
    } else if (CLOSES.includes(c)) {
      diff -= 1;
    }
  }

  return diff;
}

function getBack(line: string, pattern: Pattern, state: State, listsEnd: string[]): number {
  // Only need to dedent if indentation is present
  if (state.indent.actual === 0) {
    return 0;
  }

  if (pattern.contains_env_end && line.includes(ENV_END)) {
    // Documents get no global indentation
    if (line.includes(DOC_END)) {
      return 0;
    }
    // List environments get double indents for indenting items
    for (const r of listsEnd) {
      if (line.includes(r)) {
        return 2;
      }
    }
    return 1;
  }

  // Items get dedented
  if (pattern.contains_item && line.includes(ITEM)) {
    return 1;
  }

  return 0;
}

function getIndent(
  line: string,
  prevIndent: Indent,
  pattern: Pattern,
  state: State,
  listsBegin: string[],
  listsEnd: string[]
): Indent {
  const diff = getDiff(line, pattern, listsBegin, listsEnd);
  const back = getBack(line, pattern, state, listsEnd);

  const actual = prevIndent.actual + diff;
  const visual = Math.max(0, prevIndent.actual - back);

  return new Indent(actual, visual);
}

function calculateIndent(
  line: string,
  state: State,
  logs: Log[],
  file: string,
  args: Args,
  pattern: Pattern,
  listsBegin: string[],
  listsEnd: string[]
): Indent {
  const indent = getIndent(line, state.indent, pattern, state, listsBegin, listsEnd);

  // Update the state
  state.indent = indent;

  // Record the last line with zero indent
  if (indent.visual === 0) {
    state.linum_last_zero_indent = state.linum_new;
  }

  return indent;
}

function applyIndent(line: string, indent: Indent, args: Args, indentChar: string): string {
  if (!line.trim()) {
    return "";
  }

  const indentStr = indentChar.repeat(indent.visual * args.tabsize);
  return indentStr + line.trimStart();
}

function needsWrap(line: string, indentLength: number, args: Args): boolean {
  return args.wrap && (line.length + indentLength > args.wraplen);
}

function findWrapPoint(line: string, indentLength: number, args: Args): number | null {
  let wrapPoint: number | null = null;
  let afterChar = false;
  let prevChar: string | null = null;

  let lineWidth = 0;
  const wrapBoundary = args.wrapmin - indentLength;

  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    lineWidth += 1;
    if (lineWidth > wrapBoundary && wrapPoint !== null) {
      break;
    }
    if (c === " " && prevChar !== "\\") {
      if (afterChar) {
        wrapPoint = i;
      }
    } else if (c !== "%") {
      afterChar = true;
    }
    prevChar = c;
  }

  return wrapPoint;
}

function applyWrap(
  line: string,
  indentLength: number,
  state: State,
  file: string,
  args: Args,
  logs: Log[],
  pattern: Pattern
): [string, string, string] | null {
  if (args.verbosity >= 3) {
    logs.push({
      level: "TRACE",
      file,
      message: "Wrapping long line.",
      linum_new: state.linum_new,
      linum_old: state.linum_old,
      line
    });
  }

  const wrapPoint = findWrapPoint(line, indentLength, args);
  const commentIndex = findCommentIndex(line, pattern);

  if (wrapPoint === null || wrapPoint > args.wraplen) {
    logs.push({
      level: "WARN",
      file,
      message: "Line cannot be wrapped.",
      linum_new: state.linum_new,
      linum_old: state.linum_old,
      line
    });
    return null;
  }

  const thisLine = line.substring(0, wrapPoint);
  let nextLineStart: string;

  if (commentIndex !== null && wrapPoint > commentIndex) {
    nextLineStart = COMMENT_LINE_START;
  } else {
    nextLineStart = TEXT_LINE_START;
  }

  const nextLine = line.substring(wrapPoint + 1);

  return [thisLine, nextLineStart, nextLine];
}

function needsSplit(line: string, pattern: Pattern): boolean {
  const containsSplittableEnv = pattern.contains_splitting && RE_SPLITTING_SHARED_LINE.test(line);

  if (containsSplittableEnv) {
    const commentIndex = findCommentIndex(line, pattern);
    if (commentIndex === null) {
      return true;
    }

    const match = RE_SPLITTING_SHARED_LINE_CAPTURE.exec(line);
    // In JS, match.groups contains named groups
    if (match && match.groups && match.groups.env) {
      // We need the start index of 'env' group. 
      // JS RegExp doesn't give group indices directly unless 'd' flag is used (ES2022).
      // But we can infer it from 'prev' length.
      const envStartIndex = match.groups.prev.length;

      if (envStartIndex > commentIndex) {
        return false;
      } else {
        return true;
      }
    }
    return true;
  } else {
    return false;
  }
}

function splitLine(line: string, state: State, file: string, args: Args, logs: Log[]): [string, string] {
  const match = RE_SPLITTING_SHARED_LINE_CAPTURE.exec(line);
  if (!match || !match.groups) {
    return [line, ""];
  }

  const prev = match.groups.prev;
  const rest = match.groups.env;

  if (args.verbosity >= 3) {
    logs.push({
      level: "TRACE",
      file,
      message: "Placing environment on new line.",
      linum_new: state.linum_new,
      linum_old: state.linum_old,
      line
    });
  }

  return [prev, rest];
}

function setIgnoreAndReport(
  line: string,
  tempState: State,
  logs: Log[],
  file: string,
  pattern: Pattern
): boolean {
  tempState.ignore = getIgnore(line, tempState, logs, file, true);
  tempState.verbatim = getVerbatim(line, tempState, logs, file, true, pattern);

  return tempState.verbatim.visual || tempState.ignore.visual;
}

function cleanText(text: string, args: Args): string {
  // Remove extra newlines
  text = text.replace(RE_NEWLINES, `${LINE_END}${LINE_END}`);

  // Remove tabs if they shouldn't be used
  if (args.tabchar !== "\t") {
    text = text.replace(/\t/g, " ".repeat(args.tabsize));
  }

  // Remove trailing spaces
  // Note: JS replace with global flag replaces all occurrences
  // We need to be careful with RE_TRAIL as it matches " +\\n"
  text = text.replace(/ +\n/g, LINE_END);

  return text;
}

function removeTrailingSpaces(text: string): string {
  return text.replace(/ +\n/g, LINE_END);
}

function removeTrailingBlankLines(text: string): string {
  return text.trimEnd() + LINE_END;
}

function indentsReturnToZero(state: State): boolean {
  return state.indent.actual === 0;
}

export function formatLatex(text: string): string {
  const args = DEFAULT_ARGS;
  const file = "input.tex";
  const [formattedText] = _formatLatex(text, file, args);
  return formattedText.trim();
}

function _formatLatex(oldText: string, file: string, args: Args): [string, Log[]] {
  const logs: Log[] = [];
  logs.push({ level: "INFO", file, message: "Formatting started." });

  oldText = cleanText(oldText, args);
  const oldLines = oldText.split('\n').map((line, index) => ({ linumOld: index + 1, line }));

  // Initialize
  let state = new State();
  const queue: { linumOld: number, line: string }[] = [];
  let newText = "";

  const indentChar = args.tabchar === "\t" ? "\t" : " ";
  const listsBegin = args.lists.map(l => `\\begin{${l}}`);
  const listsEnd = args.lists.map(l => `\\end{${l}}`);

  while (true) {
    let currentItem: { linumOld: number, line: string } | undefined;

    if (queue.length > 0) {
      currentItem = queue.shift();
    } else if (oldLines.length > 0) {
      currentItem = oldLines.shift();
    } else {
      break;
    }

    if (!currentItem) break;

    let { linumOld, line } = currentItem;
    const pattern = Pattern.new(line);

    const tempState = new State({
      linum_old: linumOld,
      linum_new: state.linum_new,
      ignore: new Ignore(state.ignore.actual, state.ignore.visual),
      indent: new Indent(state.indent.actual, state.indent.visual),
      verbatim: new Verbatim(state.verbatim.actual, state.verbatim.visual),
      linum_last_zero_indent: state.linum_last_zero_indent
    });

    if (!setIgnoreAndReport(line, tempState, logs, file, pattern)) {
      if (needsSplit(line, pattern)) {
        const [thisLine, nextLine] = splitLine(line, tempState, file, args, logs);
        if (nextLine) {
          queue.unshift({ linumOld, line: nextLine });
        }
        line = thisLine;
      }

      const indent = calculateIndent(line, tempState, logs, file, args, pattern, listsBegin, listsEnd);
      const indentLength = indent.visual * args.tabsize;

      if (needsWrap(line.trimStart(), indentLength, args)) {
        const wrappedLines = applyWrap(line.trimStart(), indentLength, tempState, file, args, logs, pattern);
        if (wrappedLines) {
          const [thisLine, nextLineStart, nextLine] = wrappedLines;
          // Python does: queue.insert(0, next_line) then queue.insert(0, this_line)
          // So this_line ends up first. With unshift, we need to add in reverse order:
          queue.unshift({ linumOld, line: thisLine });
          queue.unshift({ linumOld, line: nextLineStart + nextLine });
          continue;
        }
      }

      line = applyIndent(line, indent, args, indentChar);
    }

    state = tempState;
    newText += line + LINE_END;
    state.linum_new += 1;
  }

  if (!indentsReturnToZero(state)) {
    const msg = `Indent does not return to zero. Last non-indented line is line ${state.linum_last_zero_indent}`;
    logs.push({ level: "WARN", file, message: msg });
  }

  newText = removeTrailingSpaces(newText);
  newText = removeTrailingBlankLines(newText);
  logs.push({ level: "INFO", file, message: "Formatting complete." });

  return [newText, logs];
}
